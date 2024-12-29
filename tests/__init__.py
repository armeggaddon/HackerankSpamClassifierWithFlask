from flask_testing import TestCase
import os
from spamfilter import create_app
from spamfilter.models import db, File


class BaseTestCase(TestCase):


    def create_app(self):
        app = create_app({
            'TESTING':True,
            'SQLALCHEMY_DATABASE_URI':'sqlite:///' + os.path.join(os.getcwd(),'tests/test_app.db'),
            'SQLALCHEMY_TRACK_MODIFICATIONS':False,
            'SECRET_KEY':'test',
            'TEST_DATA_DIR':os.path.join(os.getcwd(), 'tests/data'),
            'INPUT_DATA_UPLOAD_FOLDER':os.path.join(os.getcwd(), 'tests/data/inputdata'),
            'ML_MODEL_UPLOAD_FOLDER':os.path.join(os.getcwd(), 'tests/data/mlmodels')
        })

        return app