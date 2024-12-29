from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

class File(db.Model):
    '''
    Define the following three columns
    1. id - Which stores a file id 
    2. name - Which stores the file name
    3. filepath - Which stores path of file, 
    '''
        # Define the columns for the File table
    id = db.Column(db.Integer, primary_key=True)  # File id
    name = db.Column(db.String(255), nullable=False)  # File name
    filepath = db.Column(db.String(255), nullable=False)  # Path of file
    
    def __init__(self, name,filepath):
        
        self.name = name
        self.filepath= filepath

    def save(self):
        db.session.add(self)
        db.session.commit()    
    
    def __rep__(self):
        return "<File : {}>".format(self.name)
    
