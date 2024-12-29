from flask import render_template, request, flash, redirect, Blueprint, url_for
from werkzeug import secure_filename
import os, re
from flask import current_app
from spamfilter.models import db, File
import json
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from spamfilter.forms import InputForm
from spamfilter.spamclassifier import SpamClassifier



spam_api = Blueprint('SpamAPI', __name__)

def allowed_file(filename, extensions=None):
    '''
    'extensions' is either None or a list of file extensions.
    
    If a list is passed as 'extensions' argument, check if 'filename' contains 
    one of the extension provided in the list and return True or False respectively.
    
    If no list is passed to 'extensions' argument, then check if 'filename' contains
    one of the extension provided in list 'ALLOWED_EXTENSIONS', defined in 'config.py',
    and return True or False respectively.
    '''
    _, ext_ = os.path.splitext(filename)
    if extensions is not None:
        if ext_ in extensions:
            return True
        else:
            return False
    else:
        if ext_ in [".csv"]:
            return True
        else:
            return False
        
        
    

@spam_api.route('/')
def index():
    '''
    Renders 'index.html'
    '''
    return render_template('index.html')

@spam_api.route('/listfiles/<success_file>/')
@spam_api.route('/listfiles/')
def display_files(success_file=None):
    '''
    Obtain the filenames of all CSV files present in 'inputdata' folder and 
    pass it to template variable 'files'.
    
    Renders 'filelist.html' template with values  of varaibles 'files' and 'fname'.
    'fname' is set to value of 'success_file' argument.
    
    if 'success_file' value is passed, corresponding file is highlighted.
    '''
    
    input_data = os.listdir("/projects/challenge/tests/data/inputdata/")
    return render_template("fileslist.html", files=input_data, fname=success_file)



def validate_input_dataset(input_dataset_path):
    
    df_input_data = pd.read_csv(input_dataset_path)
    columns = df_input_data.columns
    input_data_col = len(columns)
    
    if input_data_col !=2:
        flash("Only 2 columns allowed: Your input csv file has {} number of columns.".format(input_data_col),"error")
        return False
    
    if ("text" not in columns) or ("spam" not in columns):
        flash('Differnt Column Names: Only column names "text" and "spam" are allowed.',"error")
        return False
    
    is_integer_column = df_input_data['spam'].apply(lambda x: isinstance(x, int)).all()
    if not is_integer_column:
        flash( "Values of spam column are not of integer type.","error")
        return False
    
    is_binary_column = df_input_data['spam'].isin([0, 1]).all()
    if not is_binary_column:
        non_binary_series = df_input_data.loc[~df_input_data['spam'].isin([0, 1]), 'spam']
        data = non_binary_series.to_list()
        my_data = ",".join(str(i) for i in data)
        flash( "Only 1 and 0 values are allowed in spam column: Unwanted values {} appear in spam column".format(my_data),"error")
        return False
        
    all_strings = df_input_data['text'].apply(lambda x: isinstance(x, str)).all()
    if not all_strings:
        flash( "Values of text column are not of string type.","error")
        return False
    
    all_start_with_subject = df_input_data['text'].str.startswith('Subject:').all()
    if not all_start_with_subject:
        flash( 'Some of the input emails does not start with keyword "Subject:".',"error")
        return False
        
    return True
    '''
        Validate the following details of an Uploaded CSV file
    
    1. The CSV file must contain only 2 columns. If not display the below error message.
    'Only 2 columns allowed: Your input csv file has '+<No_of_Columns_found>+ ' number of columns.'
    
    
    2. The column names must be "text" nad "spam" only. If not display the below error message.
    'Differnt Column Names: Only column names "text" and "spam" are allowed.'
    
    3. The 'spam' column must conatin only integers. If not display the below error message.
    'Values of spam column are not of integer type.'
    
    4. The values of 'spam' must be either 0 or 1. If not display the below error message.
    'Only 1 and 0 values are allowed in spam column: Unwanted values ' + <Unwanted values joined by comma> + ' appear in spam column'
    
    5. The 'text' column must contain string values. If not display the below error message.
    'Values of text column are not of string type.'
    
        6. Every input email must start with 'Subject:' pattern. If not display the below error message.
    'Some of the input emails does not start with keyword "Subject:".'
    
    Return True if all 6 validations pass.
    '''

@spam_api.route('/upload/', methods=['GET', 'POST'])
def file_upload():
    
    if request.method == "GET":
        return render_template("upload.html")
        
    elif request.method == "POST":
        file = request.files.get('uploadfile')
        print("@@@@@@@@@@@@@file@@@@@@@@@@",file)
        if file is None:
            flash("No file part")
            return render_template("upload.html")

        flag = allowed_file(file.filename)
        if not flag:
            flash("Only CSV Files are allowed as Input.","error")
            return render_template("upload.html")
        else:
            filename = secure_filename(file.filename)
            # Save the file to the configured upload folder
            file_path = os.path.join("/projects/challenge/tests/data/inputdata", filename)
            file.save(file_path)
            file_flag = validate_input_dataset(file_path)
            if not file_flag:
                os.remove(file_path)
                return render_template("upload.html")
            else:
                f_model = File(filename,file_path)
                f_model.save()
                return redirect(url_for('SpamAPI.display_files',success_file=file.filename))
            
    
    '''
    If request is GET, Render 'upload.html'
    
    If request is POST, capture the uploaded file a
    
    check if the uploaded file is 'csv' extension, using 'allowed_file' defined above.
    
    if 'allowed_file' returns False, display the below error message and redirect to 'upload.html' with GET request.
    'Only CSV Files are allowed as Input.'
    
    if 'allowed_file' returns True, save the file in 'inputdata' folder and 
    validate the uploaded csv file using 'validate_input_dataset' defined above.
    
    if 'validate_input_dataset' returns 'False', remove the file from 'inputdata' folder,
    redirect to 'upload.html' with GET request and respective error message.
    
    if 'validate_input_dataset' returns 'True', create a 'File' object and save it in database, and
    render 'display_files' template with template varaible 'success_file', set to filename of uploaded file.
    
    '''
    


def validate_input_text(intext):
    
    space_flag = True
    od = OrderedDict()
    for data in intext.splitlines():
        if len(data) != 0:
            if not space_flag:
                return False
            space_flag = False
            if not data.startswith("Subject:"):
                return False
            else:
                key_ = data[:30]
                od[key_]=data
        elif len(data) == 0:
            space_flag = True
    
    return od
    
        
        
    
        
    '''
    Validate the following details of input email text, provided for prediction.
    
    1. If the input email text contains more than one mail, they must be separated by atleast one blank line.
    
    2. Every input email must start with 'Subject:' pattern.
    
    Return False if any of the two validations fail.
    
    If all valiadtions pass, Return an Ordered Dicitionary, whose keys are first 30 characters of each
    input email and values being the complete email text.
    '''

    

@spam_api.route('/models/<success_model>/')
@spam_api.route('/models/')
def display_models(success_model=None):
    
    input_data = os.listdir("/projects/challenge/tests/data/mlmodels/")            
    new_input_data = []
    for data in input_data:
        if 'word_features' not in data:
            new_input_data.append(data)
    
            
    return render_template("modelslist.html",files=new_input_data, model_name=success_model)
    
    '''
    Obtain the filenames of all machine learning models present in 'mlmodels' folder and 
    pass it to template variable 'files'.
    
    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.
    
    Consider only the model and not the word_features.pk files.
    
    Renders 'modelslist.html' template with values  of varaibles 'files' and 'model_name'.
    'model_name' is set to value of 'success_model' argument.
    
    if 'success_model value is passed, corresponding model file name is highlighted.
    '''



def isFloat(value):
    '''
    Return True if <value> is a float, else return False
    '''
    return isinstance(value, float)

def isInt(value):
    '''
    Return True if <value> is an integer, else return False
    '''
    return isinstance(value, int)
    

@spam_api.route('/train/', methods=['GET', 'POST'])
def train_dataset():
    
    input_data = os.listdir("/projects/challenge/tests/data/inputdata/")
    
        # Filter for CSV files
    csv_files = [f for f in input_data if f.endswith('.csv')]
    
    if request.method == "GET":
        return render_template("train.html",train_files=csv_files)
        
    elif request.method == "POST":
        train_file = request.form.get("train_file")
        train_size = request.form.get("train_size")
        random_state = request.form.get("random_state")
        shuffle = request.form.get("shuffle")
        stratify = request.form.get("stratify")
        
        if train_file is None:
            print("################train_file######################",train_file)
            flash("No CSV file is selected","error")
            return render_template("train.html",train_files=csv_files)
        if train_size is None:
            print("################train_size######################",train_size)
            flash("No value provided for size of training data set.","error")
            return render_template("train.html",train_files=csv_files)
        
        
        if train_size.isalpha():
            pass
        else:
            train_size = eval(train_size)
        if not isinstance(train_size,float):
            print("################error######################","error")
            flash("Training Data Set Size must be a float.","error")
            return render_template("train.html",train_files=csv_files)
        if not ((train_size >= 0.0) and (train_size <=1.0)):
            print("################train_size######################",train_size)
            flash("Training Data Set Size Value must be in between 0.0 and 1.0","error")
            return render_template("train.html",train_files=csv_files)
        if random_state is None:
            print("################-random_state######################",random_state)
            flash("No value provided for random state.","error")
            return render_template("train.html",train_files=csv_files)
        if random_state.isalpha():
            pass
        else:
            random_state = eval(random_state)            
        if not isinstance(random_state,int):
            print("################random_state######################",random_state)
            flash("Random State must be an integer.","error")
            return render_template("train.html",train_files=csv_files) 
        
        if shuffle is None:
            print("################shuffle######################",shuffle)
            flash("No option for shuffle is selected.","error")
            return render_template("train.html",train_files=csv_files)
        
        if shuffle == "N":
            if stratify == "Y":
                flash("When Shuffle is No, Startify cannot be Yes.","error")
                return render_template("train.html",train_files=csv_files)

        fn, ext = os.path.splitext(os.path.basename(train_file))
        data = pd.read_csv("/projects/challenge/tests/data/inputdata/"+train_file) 

        if stratify == "N":
            stratify = None
        if shuffle == "N":
            shu = False 
            stratify = None  
        elif shuffle == "Y":
            shu = True
            if stratify != "N":
                stratify = data['spam'].values
            
        try:
            train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values,
                                                                    data["spam"].values,
                                                                    train_size=train_size,
                                                                    random_state = random_state,
                                                                    shuffle = shu,
                                                                    stratify=stratify)        
        except Exception as e:
            print("###############meow meow#########################",e)
        classifier = SpamClassifier()
        classifier_model, model_word_features = classifier.train(train_X, train_Y)
        model_name = '{}.pk'.format(fn)
        model_word_features_name = '{}_word_features.pk'.format(fn)
        with open("/projects/challenge/tests/data/mlmodels/{}".format(model_name), 'wb') as model_fp:
            pickle.dump(classifier_model, model_fp)
        with open("/projects/challenge/tests/data/mlmodels/{}".format(model_word_features_name), 'wb') as model_fp:
                pickle.dump(model_word_features, model_fp)
        print('DONE')
        return redirect(url_for('SpamAPI.display_models',success_model=model_name))
    
    '''
    If request is of GET method, render 'train.html' template with tempalte variable 'train_files',
    set to list if csv files present in 'inputdata' folder.
    
    If request is of POST method, capture values associated with
    'train_file', 'train_size', 'random_state', and 'shuffle'
    
    if no 'train_file' is selected, render the same page with GET Request and below error message.
    'No CSV file is selected'
    
    if 'train_size' is None, render the same page with GET Request and below error message.
    'No value provided for size of training data set.'
    
    if 'train_size' value is not float, render the same page with GET Request and below error message.
    'Training Data Set Size must be a float.
    
    if 'train_size' value is not in between 0.0 and 1.0, render the same page with GET Request and below error message.
    'Training Data Set Size Value must be in between 0.0 and 1.0' 
    
    if 'random_state' is None,render the same page with GET Request and below error message.
    'No value provided for random state.''
    
    if 'random_state' value is not an integer, render the same page with GET Request and below error message.
    'Random State must be an integer.'
    
    
    if 'shuffle' is None, render the same page with GET Request and below error message.
    'No option for shuffle is selected.'
    
    if 'shuffle' is set to 'No' when 'Startify' is set to 'Yes', render the same page with GET Request and below error message.
    'When Shuffle is No, Startify cannot be Yes.'
    
    If all input values are valid, build the model using submitted paramters and methods defined in
    'spamclassifier.py' and save the model and model word features file in 'mlmodels' folder.
    
    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.
    
    Finally render, 'display_models' template with value of template varaible 'success_model' 
    set to name of model generated, ie. 'sample.pk'
    '''

    
@spam_api.route('/results/')
def display_results():
    
    with open('predictions.json', 'r') as file:
        data = json.load(file)
    return render_template("displayresults.html",predictions=data.items())
    '''
    Read the contents of 'predictions.json' and pass those values to 'predictions' template varaible
    
    Render 'displayresults.html' with value of 'predictions' template variable.
    '''
    
    
@spam_api.route('/predict/', methods=['GET', "POST"])
def predict():
    input_model = os.listdir("/projects/challenge/tests/data/mlmodels/")
    model_list = []
    for model_ in input_model:
        fn, ext = os.path.splitext(model_)
        model_list.append(fn)
    if request.method == "GET":
        input_model = os.listdir("/projects/challenge/tests/data/mlmodels/")
        model_list = []
        for model_ in input_model:
            fn, ext = os.path.splitext(model_)
            model_list.append(fn)
        return render_template("emailsubmit.html",form=InputForm(model_list))
        
    if request.method == "POST":
        
        inputemail = request.form.get("inputemail")
        inputfile = request.files.get('inputfile')
        inputmodel = request.form.get("inputmodel")
        print("##############inputemail",type(inputemail))
        print("##############inputfile",inputfile)
        print("##############inputmodel",inputmodel)
        
        if inputemail is None and inputfile is None:
            print("###############inputemail and inputfile is none##########")
            flash("No Input: Provide a Single or Multiple Emails as Input.","error")
            return render_template("emailsubmit.html",form=InputForm(model_list))
            
        if inputemail and inputfile and inputfile.filename != '':
            print("###############two inputs##########")
            # Flash an error message
            flash('Two Inputs Provided: Provide Only One Input.', 'error')
            # Re-render the same page with GET request logic
            return render_template("emailsubmit.html",form=InputForm(model_list))
            
        if inputfile and inputfile.filename.endswith('.txt'):
            print("###############file_read##############")
            inputfile.save("/projects/challenge/tests/data/inputdata/"+inputfile.filename)
            with open("/projects/challenge/tests/data/inputdata/"+inputfile.filename, 'r') as file:
                input_txt = file.read()
            
        if inputemail is not None:
            input_txt = inputemail

        output_ = validate_input_text(input_txt.strip())
        
        if output_ is False:
            print("###############Unexpected format##########")
            flash('Unexpected Format : Input Text is not in Specified Format.', 'error')
            # Re-render the same page with GET request logic
            return render_template("emailsubmit.html",form=InputForm(model_list)) 
        
        if inputmodel is None:
            print("###############single model##########")
            flash('Please Choose a single Model', 'error')
            return render_template("emailsubmit.html",form=InputForm(model_list)) 
        else:
            
            sc = SpamClassifier()
            sc.load_model(inputmodel)
            output = sc.predict(output_)
            print("#############CATT############")
            print(output)
            od = OrderedDict()
            for key,value in output.items():
                if value == 0:
                    value_ = "SPAM"
                elif value ==1:
                    value_ = "SPAM"
                od[key]=value_
            with open("predictions.json", "w") as outfile: 
                json.dump(od, outfile)
            return redirect(url_for('SpamAPI.display_results'))        
                       
    '''
    If request is of GET method, render 'emailsubmit.html' template with value of template
    variable 'form' set to instance of 'InputForm'(defined in 'forms.py'). 
    Set the 'inputmodel' choices to names of models (in 'mlmodels' folder), with out extension i.e .pk
    
    If request is of POST method, perform the below checks
    
    1. If input emails is not provided either in text area or as a '.txt' file, render the same page with GET Request and below error message.
    'No Input: Provide a Single or Multiple Emails as Input.' 
    
    2. If input is provided both in text area and as a file, render the same page with GET Request and below error message.
    'Two Inputs Provided: Provide Only One Input.'
    
    3. In case if input is provided as a '.txt' file, save the uploaded file into 'inputdata' folder and read the
     contents of file into a variable 'input_txt'
    
    4. If input provided in text area, capture the contents in the same variable 'input_txt'.
    
    5. validate 'input_txt', using 'validate_input_text' function defined above.
    
    6. If 'validate_input_text' returns False, render the same page with GET Request and below error message.
    'Unexpected Format : Input Text is not in Specified Format.'

    
    7. If 'validate_input_text' returns a Ordered dictionary, choose a model and perform prediction of each input email using 'predict' method defined in 'spamclassifier.py'
    
    8. If no input model is choosen, render the same page with GET Request and below error message.
    'Please Choose a single Model'
    
    9. Convert the ordered dictionary of predictions, with 0 and 1 values, to another ordered dictionary with values 'NOT SPAM' and 'SPAM' respectively.
    
    10. Save thus obtained predictions ordered dictionary into 'predictions.json' file.
    
    11. Render the template 'display_results'
    
    '''
    
