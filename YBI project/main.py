# main.py

from flask import Flask, render_template
from models.model1.app import model1_bp
from models.model2.app import model2_bp
from models.model3.app import model3_bp
from models.model4.app import model4_bp
from models.model5.app import model5_bp
from models.model6.app import model6_bp
from models.model7.app import model7_bp
from models.model8.app import model8_bp
# Import other model blueprints similarly

app = Flask(__name__)

# Register blueprints for each model
app.register_blueprint(model1_bp, url_prefix='/model1')
app.register_blueprint(model2_bp, url_prefix='/model2')
app.register_blueprint(model3_bp, url_prefix='/model3')
app.register_blueprint(model4_bp, url_prefix='/model4')
app.register_blueprint(model5_bp, url_prefix='/model5')
app.register_blueprint(model6_bp, url_prefix='/model6')
app.register_blueprint(model7_bp, url_prefix='/model7')
app.register_blueprint(model8_bp, url_prefix='/model8')
# Register other model blueprints here similarly

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
