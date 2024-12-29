from wtforms import Form, TextAreaField, SubmitField, FileField, validators, RadioField

    
class InputForm(Form):
    """
    Form with an email text area, file upload, radio buttons, and a submit button.
        Include 4 fields : 1. inputemail - a Text Area Field
                       2. inputfile - a File Field
                       3. inputmodel - a Radio Button
                       4. submit - a Submit Button
    """
    # Email text area field
    inputemail = TextAreaField('Email', [validators.DataRequired(message="Email is required")])
    
    # File upload field
    inputfile = FileField('File',[validators.DataRequired(message="File is required")])
    
    # Radio button field
    inputmodel = RadioField('Model Choice', validators=[validators.DataRequired(message="Please select a model")]
    )
    
    # Submit button
    submit = SubmitField('Submit')

    def __init__(self, model_choices=None, *args, **kwargs):
        """
        Initialize the form with dynamic model choices.
        """
        super(InputForm, self).__init__(*args, **kwargs)
        if model_choices:
            self.inputmodel.choices = model_choices