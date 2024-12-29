from tests import BaseTestCase
import os
import re
from flask import current_app
from shutil import copyfile
from spamfilter.models import File, db
import unittest

class TestPage(BaseTestCase):

    def test_get_url_status(self):
        response = self.client.get('/')
        self.assertEqual(200, response.status_code)

    def test_template_used(self):
        response = self.client.get('/')
        self.assertTemplateUsed('index.html')

    def test_get_response(self):
        response = self.client.get('/')
        self.assertIn(
            b'Welcome to SPAM FILTER application. This application can be used to perform the following tasks.', response.data)

    def test_get_url_status1(self):
        response = self.client.get('/upload/')
        self.assertEqual(200, response.status_code)

    def test_template_used1(self):
        response = self.client.get('/upload/')
        self.assertTemplateUsed('upload.html')

    def test_get_response1(self):
        response = self.client.get('/upload/')
        self.assertIn(b'Uploading a Data Set', response.data)
        self.assertIn(
            b'<input type = "file" name = "uploadfile" />', response.data)
        self.assertIn(b'<input type="reset" value="Refresh">', response.data)
        self.assertIn(b'<input type = "submit"/>', response.data)

    def test_post_url_response1(self):
        response = self.client.post('/upload/',
                                    data=dict(
                                        uploadfile=(os.path.join(current_app.config['TEST_DATA_DIR'], 'sample_emails.csv'),
                                                    'sample_emails.csv')
                                    ),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)

        test_file = File.query.filter(File.name == 'sample_emails.csv').first()
        self.assertIn(
            b'File : <span style="color:green">sample_emails.csv</span> is uploaded successfully', response.data)
        self.assertEqual('sample_emails.csv', test_file.name)
        File.query.filter(File.name == 'sample_emails.csv').delete()
        db.session.commit()

    def test_post_url_response2(self):
        response = self.client.post('/upload/',
                                    data=dict(
                                        uploadfile=os.path.join(
                                            current_app.config['TEST_DATA_DIR'], 'sample_emails.csv')
                                    ),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)

        self.assertIn(b'No file part', response.data)

    def test_post_url_response3(self):
        response = self.client.post('/upload/',
                                    data=dict(
                                        uploadfile=(os.path.join(current_app.config['TEST_DATA_DIR'], 'sample_email3.txt'),
                                                    'sample_email3.txt')
                                    ),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)

        self.assertIn(b'Only CSV Files are allowed as Input', response.data)

    @classmethod
    def setUpClass(cls):
        model_file = os.path.join(
            os.getcwd(), 'tests/data/mlmodels/sample_emails.pk')
        model_word_features_file = os.path.join(
            os.getcwd(), 'tests/data/mlmodels/sample_emails_word_features.pk')
        if os.path.exists(model_file):
            os.remove(model_file)
        if os.path.exists(model_word_features_file):
            os.remove(model_word_features_file)
        if os.path.exists(os.path.join(os.getcwd(), 'tests/data/sample_emails.csv')):
            copyfile(os.path.join(os.getcwd(), 'tests/data/sample_emails.csv'),
                     os.path.join(os.getcwd(), 'tests/data/inputdata/sample_emails.csv'))

    @classmethod
    def tearDownClass(cls):
        model_file = os.path.join(
            os.getcwd(), 'tests/data/mlmodels/sample_emails.pk')
        model_word_features_file = os.path.join(
            os.getcwd(), 'tests/data/mlmodels/sample_emails_word_features.pk')
        input_file = os.path.join(
            os.getcwd(), 'tests/data/inputdata/sample_emails.csv')
        if os.path.exists(model_file):
            os.remove(model_file)
        if os.path.exists(model_word_features_file):
            os.remove(model_word_features_file)
        if os.path.exists(input_file):
            os.remove(input_file)

    def test_get_url_status2(self):
        response = self.client.get('/train/')
        self.assertEqual(200, response.status_code)

    def test_template_used2(self):
        response = self.client.get('/train/')
        self.assertTemplateUsed('train.html')

    def test_get_response2(self):
        response = self.client.get('/train/')
        self.assertIn(b'Train a Data Set', response.data)
        self.assertIn(
            b'<input type="radio" name="train_file" value=sample_emails.csv>', response.data)
        self.assertIn(
            b'<input type="text" name="train_size" value="0.80"/>', response.data)
        self.assertIn(
            b'<input type="text" name="random_state" value="50"/>', response.data)
        self.assertIn(
            b'<input type="radio" name="shuffle" value="Y" checked/>', response.data)
        self.assertIn(
            b'<input type="radio" name="shuffle" value="N"/>', response.data)
        self.assertIn(
            b'<input type="radio" name="stratify" value="N" checked/>', response.data)
        self.assertIn(
            b'<input type="radio" name="stratify" value="Y" />', response.data)

    def test_post_response1(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_file='sample_emails.csv',
                                        train_size=0.80,
                                        random_state=50,
                                        shuffle='Y',
                                        stratify='N',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(
            b'Model : <span style="color:green">sample_emails.pk</span> is successfully created.', response.data)
        self.assertTrue(os.path.exists(os.path.join(
            current_app.config['ML_MODEL_UPLOAD_FOLDER'], 'sample_emails.pk')))
        self.assertTrue(os.path.exists(os.path.join(
            current_app.config['ML_MODEL_UPLOAD_FOLDER'], 'sample_emails_word_features.pk')))

    def test_post_response2(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_size=0.80,
                                        random_state=50,
                                        shuffle='Y',
                                        stratify='N',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(b'No CSV file is selected', response.data)

    def test_post_response3(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_file='sample_emails.csv',
                                        random_state=50,
                                        shuffle='Y',
                                        stratify='N',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(
            b'No value provided for size of training data set.', response.data)

    def test_post_response4(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_file='sample_emails.csv',
                                        train_size='hello',
                                        random_state=50,
                                        shuffle='Y',
                                        stratify='N',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(
            b'Training Data Set Size must be a float.', response.data)

    def test_post_response5(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_file='sample_emails.csv',
                                        train_size=2.5,
                                        random_state=50,
                                        shuffle='Y',
                                        stratify='N',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(
            b'Training Data Set Size Value must be in between 0.0 and 1.0', response.data)

    def test_post_response6(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_file='sample_emails.csv',
                                        train_size=0.8,
                                        shuffle='Y',
                                        stratify='N',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(b'No value provided for random state.', response.data)

    def test_post_response7(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_file='sample_emails.csv',
                                        train_size=0.8,
                                        random_state='hello',
                                        shuffle='Y',
                                        stratify='N',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(b'Random State must be an integer.', response.data)

    def test_post_response8(self):
        response = self.client.post('/train/',
                                    data=dict(
                                        train_file='sample_emails.csv',
                                        train_size=0.8,
                                        random_state=50,
                                        shuffle='N',
                                        stratify='Y',
                                    ),
                                    follow_redirects=True
                                    )
        self.assertIn(
            b'When Shuffle is No, Startify cannot be Yes.', response.data)

    @classmethod
    def setUpClass(cls):
        if os.path.exists(os.path.join(os.getcwd(), 'tests/data/sample_emails.pk')):
            copyfile(os.path.join(os.getcwd(), 'tests/data/sample_emails.pk'),
                     os.path.join(os.getcwd(), 'tests/data/mlmodels/sample_emails.pk'))
        if os.path.exists(os.path.join(os.getcwd(), 'tests/data/sample_emails_word_features.pk')):
            copyfile(os.path.join(os.getcwd(), 'tests/data/sample_emails_word_features.pk'),
                     os.path.join(os.getcwd(), 'tests/data/mlmodels/sample_emails_word_features.pk'))
        if os.path.exists(os.path.join(os.getcwd(), 'tests/data/sample_email3.txt')):
            copyfile(os.path.join(os.getcwd(), 'tests/data/sample_email3.txt'),
                     os.path.join(os.getcwd(), 'tests/data/inputdata/sample_email3.txt'))

        cls.test_emailtext = '''Subject: fwd : next tuesday at 9 am  for  immediate release  cal - bay ( stock symbol : cbyi )  watch for analyst " strong buy recommendations " and several advisory  newsletters picking cbyi . cbyi has filed to be traded on  theotcbb , share prices historically increase when companies get  listed on this larger tradingexhange . cbyi is trading around $ . 30 ?  and should skyrocket to $ 2 . 66 - $ 3 . 25 a share in the near future .  put cbyi on your watch list , acquire a postion  today .  reasons to invest in cbyi  a profitable company , no debt and is on track to beat all earnings  estimates with increased revenue of 50 % annually !  one of the fastest growing distributors in environmental safety  equipment instruments .  excellent management team , several exclusive  contracts . impressive client list including theu . s . air force ,  anheuser - busch , chevron refining and mitsubishi heavy industries ,  ge - energy environmental research .  rapidly growing industry  industry revenues exceed $ 900 million , estimates indicate that there  could be as much as $ 25 billion from " smell technology " by the end of  2003 .  ! ! ! ! congratulations  ! ! ! ! ! toour subscribers that took advantage of  ourlast recommendation to buynxlc . it rallied from $ 7 . 87  to $ 11 . 73 !  all removes honered . please allow 7  days to be removed and send all address to : honey 9531 @ mail . net . cn  certain statements contained in this news release may be  forward - looking statements within the meaning of the private securities  litigation reform act of 1995 . these statements may be identified by such terms  as " expect " , " believe " , " may " , " will " , and " intend " or similar terms . we are not  a registered investment advisor or a broker dealer . this is not an offer to buy  or sell securities . no recommendation that the securities of the companies  profiled should be purchased , sold or held by individuals or entities that learn  of the profiled companies . we were paid $ 27 , 000 in cash by a third party to  publish this report . investing in companies profiled is high - risk and use of  this information is for reading purposes only . if anyone decides to act as an  investor , then it will be that investor ' s sole risk . investors are advised not  to invest without the proper advisement from an attorney or a registered  financial broker . do not rely solely on the information presented , do additional  independent research to form your own opinion and decision regarding investing  in the profiled companies . be advised that the purchase of such high - risk  securities may result in the loss of your entire investment . the owners of this publication may already own free trading shares in  cbyi and may immediately sell all or a portion of these shares into the open  market at or about the time this report is published . factual statements  are made as of the date stated and are subject to change without notice .  not intended for recipients or residents of ca , co , ct , de , id , il , ia , la , mo , nv , nc , ok , oh , pa , ri , tn , va , wa , wv , wi . void where  prohibited . copyright c 2001  * * * * *

Subject: visit may 4 th  vince :  per susan ' s email below , do you want to go to the luncheon for john  hennessey ? she doesn ' t say where the lunch is going to be , did you  get an invite ?  the only thing you have that day is 9 : 00 am larry thorne and the  energy derivatives class at 11 : 30 .  let me know .  thanks !  shirley  - - - - - - - - - - - - - - - - - - - - - - forwarded by shirley crenshaw / hou / ect on 04 / 17 / 2001 11 : 55 am - - - - - - - - - - - - - - - - - - - - - - - - - - -  " susan c . hansen " on 04 / 17 / 2001 10 : 47 : 38 am  to : shirley . crenshaw @ enron . com  cc : clovell @ stanford . edu , donna lawrence  subject : visit may 4 th  hi shirley ,  thanks for corresponding with carol during my absence , and confirming our  meeting with vince kaminski at 1 : 30 on may 4 th . i have a question about  the logistics . i believe dr . kaminski has received an invitation to an  event in houston : new stanford president john hennessy is visiting a number  of cities on a " welcome tour , " and it just so happens he is hosting a  luncheon in houston on may 4 th . if dr . kaminski wants to attend the  hennessy welcome tour luncheon , donna lawrence and i could meet with him at  1 : 30 somewhere in the hotel . if he ' s not attending the presidential event ,  please let me know where you are located , and we ' ll plan travel time  accordingly .  regards ,  susan  susan c . hansen  director , corporate relations  school of engineering  stanford university  stanford , ca 94305 - 4027  ( 650 ) 725 - 4219
        '''

        pattern1 = re.compile(r'[\r\n][\r\n]+', re.DOTALL)

        inemails = pattern1.split(cls.test_emailtext)

        cls.modified_test_emailtext = ['Hello ' + email for email in inemails]

    @classmethod
    def tearDownClass(cls):
        model_file = os.path.join(
            os.getcwd(), 'tests/data/mlmodels/sample_emails.pk')
        model_word_features_file = os.path.join(
            os.getcwd(), 'tests/data/mlmodels/sample_emails_word_features.pk')
        inputemail_file = os.path.join(
            os.getcwd(), 'tests/data/inputdata/sample_email3.txt')
        if os.path.exists(model_file):
            os.remove(model_file)
        if os.path.exists(model_word_features_file):
            os.remove(model_word_features_file)
        if os.path.exists(inputemail_file):
            os.remove(inputemail_file)

    def test_template_used3(self):
        response = self.client.get('/predict/')
        self.assertTemplateUsed('emailsubmit.html')

    def test_post_response11(self):
        response = self.client.post('/predict/',
                                    data=dict(inputmodel='sample_emails'),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)
        self.assertIn(
            b'No Input: Provide a Single or Multiple Emails as Input.', response.data)

    def test_post_response21(self):
        response = self.client.post('/predict/',
                                    data=dict(inputmodel='sample_emails',
                                              inputfile=(os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'],
                                                                      'sample_email3.txt'),
                                                         'sample_email3.txt'),
                                              inputemail=self.test_emailtext),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)
        self.assertIn(
            b'Two Inputs Provided: Provide Only One Input.', response.data)

    def test_post_response31(self):
        response = self.client.post('/predict/',
                                    data=dict(
                                        inputfile=(os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'],
                                                                'sample_email3.txt'),
                                                   'sample_email3.txt'),
                                    ),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)
        self.assertIn(b'Please Choose a single Model', response.data)

    def test_post_response41(self):
        response = self.client.post('/predict/',
                                    data=dict(
                                        inputemail=self.modified_test_emailtext,
                                        inputmodel='sample_emails'
                                    ),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)
        self.assertIn(
            b'Unexpected Format : Input Text is not in Specified Format.', response.data)

    def test_post_response51(self):
        response = self.client.post('/predict/',
                                    data=dict(
                                        inputemail=self.test_emailtext,
                                        inputmodel='sample_emails'
                                    ),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)
        self.assertIn(b'Prediction Results', response.data)
        self.assertEqual(2, response.data.count(
            b'<span style="color:red;font-size:20px;font-weight:bold;">SPAM</span>'))

    def test_post_response61(self):
        response = self.client.post('/predict/',
                                    data=dict(
                                        inputfile=(os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'],
                                                                'sample_email3.txt'),
                                                   'sample_email3.txt'),
                                        inputmodel='sample_emails'
                                    ),
                                    content_type='multipart/form-data',
                                    follow_redirects=True)
        self.assertIn(b'Prediction Results', response.data)
        self.assertEqual(3, response.data.count(
            b'<span style="color:red;font-size:20px;font-weight:bold;">SPAM</span>'))

    def test_config(self):
        self.assertTrue(self.app.testing)

    def test_appfactory_home(self):
        response = self.client.get('/home')
        self.assertIn(
            response.data, b'This verifies Application working Status : test')
        self.assertEqual(self.app, self.app)  # <Flask 'myflaskproject'>
        self.assertEqual(self.app, self.app)


if __name__ == '__main__':
    import xmlrunner
    with open('test-reports/results.xml', 'wb') as output:
        unittest.main(
            testRunner=xmlrunner.XMLTestRunner(output=output),
            failfast=False, buffer=False, catchbreak=False)
