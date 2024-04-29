from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import jwt
from werkzeug.security import check_password_hash, generate_password_hash
import pickle
import torch
from transformers import BertTokenizer

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'bloom_bert_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, origins=["http://localhost:3000"])  # Allow requests only from http://localhost:3000

db = SQLAlchemy(app)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    creation_date = db.Column(db.DateTime, nullable=False, default=datetime.now)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)


class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    creation_date = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    questions = db.relationship('Question', backref='test', lazy='dynamic', cascade='all, delete-orphan')


class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    creation_date = db.Column(db.DateTime, default=datetime.now)

    user = db.relationship('User', backref=db.backref('feedbacks', lazy=True))


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['user_id']).first()
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)

    return decorated


# User Authentication
@app.route('/login', methods=['POST'])
def login():
    print(request.json)
    auth = request.json
    # if not auth or not auth.username or not auth.password:
    #     print(auth)
    #     return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})
    user = User.query.filter_by(username=auth['username']).first()
    if not user:
        return make_response('User not found', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})
    if check_password_hash(user.password, auth['password']):
        token = jwt.encode({'user_id': user.id, 'exp': datetime.now() + timedelta(minutes=30)},
                           app.config['SECRET_KEY'])
        return jsonify({'token': token})
    return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'message': 'Missing information!'}), 400

    user = User.query.filter((User.username == username) | (User.email == email)).first()
    if user:
        return jsonify({'message': 'User already exists!'}), 409

    new_user = User(username=username, email=email)
    new_user.set_password(password)  # Hash password

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully!'}), 201


@app.route('/tests', methods=['POST'])
@token_required
def create_test(current_user):
    data = request.get_json()
    title = data.get('title')
    questions = data.get('questions', [])  # This expects a list of question strings

    new_test = Test(title=title, user_id=current_user.id)
    db.session.add(new_test)
    db.session.commit()

    for question_content in questions:
        new_question = Question(content=question_content, test_id=new_test.id)  # Set test_id here
        db.session.add(new_question)

    db.session.commit()

    return jsonify({'message': 'Test created successfully', 'test_id': new_test.id}), 201


@app.route('/my-tests', methods=['GET'])
@token_required
def get_user_tests(current_user):
    tests = Test.query.filter_by(user_id=current_user.id).all()
    user_tests = []
    for test in tests:
        test_data = {
            'id': test.id,
            'title': test.title,
            'questions': [question.content for question in test.questions]
        }
        user_tests.append(test_data)
    return jsonify(user_tests)


@app.route('/my-tests/<int:test_id>', methods=['GET'])
@token_required
def get_user_test_by_id(current_user, test_id):
    test = Test.query.filter_by(id=test_id, user_id=current_user.id).first()
    if not test:
        return jsonify({'message': 'Test not found'}), 404

    return jsonify({
        'id': test.id,
        'title': test.title,
        'questions': [question.content for question in test.questions]
    }), 200


@app.route('/tests/<int:test_id>', methods=['PUT'])
@token_required
def update_test(current_user, test_id):
    test = Test.query.filter_by(id=test_id, user_id=current_user.id).first()
    if not test:
        return jsonify({'message': 'Test not found'}), 404

    data = request.get_json()
    test.title = data.get('title', test.title)
    if 'questions' in data:
        test.questions = [Question(content=question, test_id=test_id) for question in data['questions']]
    db.session.commit()

    return jsonify({'message': 'Test updated successfully'}), 200


@app.route('/tests/<int:test_id>/analyze', methods=['GET'])
@token_required
def analyze_test(current_user, test_id):
    test = Test.query.filter_by(id=test_id, user_id=current_user.id).first()
    if not test:
        return jsonify({'message': 'Test not found'}), 404

    classifications = [q.classification for q in test.questions]
    analysis = {classification: classifications.count(classification) for classification in set(classifications)}

    return jsonify({'test_id': test_id, 'analysis': analysis}), 200


@app.route('/tests/<int:test_id>', methods=['DELETE'])
@token_required
def delete_test(current_user, test_id):
    test = Test.query.filter_by(id=test_id, user_id=current_user.id).first()
    if not test:
        return jsonify({'message': 'Test not found'}), 404

    db.session.delete(test)
    db.session.commit()

    return jsonify({'message': 'Test deleted successfully'}), 200


# def calculate_insights_for_user(user_id):
#     # Fetch all tests created by the user
#     tests = Test.query.filter_by(user_id=user_id).all()
#
#     # Create a dictionary to hold the count of each classification category
#     classification_count = {
#         'Remembering': 0,
#         'Understanding': 0,
#         'Applying': 0,
#         'Analyzing': 0,
#         'Evaluating': 0,
#         'Creating': 0
#     }
#
#     # Iterate over each test and each question to populate the classification_count dictionary
#     for test in tests:
#         for question in test.questions:
#             if question.classification in classification_count:
#                 classification_count[question.classification] += 1
#
#     # Calculate total questions and percentages for each category
#     total_questions = sum(classification_count.values())
#     classification_percentages = {}
#     if total_questions > 0:
#         for classification, count in classification_count.items():
#             percentage = (count / total_questions) * 100
#             classification_percentages[classification] = round(percentage, 2)
#
#     return {
#         'total_questions': total_questions,
#         'counts': classification_count,
#         'percentages': classification_percentages
#     }


@app.route('/insights', methods=['GET'])
@token_required
def get_insights(current_user):
    # Fetch all tests created by the user
    tests = Test.query.filter_by(user_id=current_user.id).all()

    # Create a dictionary to hold the count of each classification category
    classification_count = {
        'Knowledge': 0,
        'Comprehension': 0,
        'Application': 0,
        'Analysis': 0,
        'Evaluation': 0,
        'Synthesis': 0
    }

    # Load the model and tokenizer
    model_path = 'ai/model/fine_tuned_bert_model.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the label encoder to decode numerical labels to category names
    label_encoder_path = 'ai/model/label_encoder.pkl'
    with open(label_encoder_path, 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    total_questions = 0  # Initialize total questions counter

    # Iterate over each test and classify questions
    for test in tests:
        questions = [question.content for question in test.questions]
        total_questions += len(questions)  # Increment total questions counter
        # Process and classify questions
        model.eval()  # Set the model to evaluation mode
        device = torch.device("mps" if torch.has_mps else "cpu")
        with torch.no_grad():  # Disable gradient computation for inference
            for question in questions:
                inputs = tokenizer.encode_plus(
                    question,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=512
                ).to(device)
                output = model(**inputs)
                prediction = torch.argmax(output.logits, dim=-1).item()  # Extract the class with the highest probability
                # Map the numerical label to the actual category
                category = label_encoder.inverse_transform([prediction])[0]
                # Update classification count dictionary
                if category in classification_count:
                    classification_count[category] += 1

    # Calculate percentages for each category only if total_questions is not zero
    classification_percentages = {category: round(count / total_questions * 100, 1) if total_questions != 0 else 0
                                  for category, count in classification_count.items()}

    # Debugging: Print classification counts and percentages
    print("Classification Counts:", classification_count)
    print("Classification Percentages:", classification_percentages)

    return jsonify({
        'total_questions': total_questions,
        'total_tests': len(tests),
        'counts': classification_count,
        'percentages': classification_percentages
    })


@app.route('/classify', methods=['POST'])
# @token_required
def classify_question():
    data = request.json
    questions = data.get('questions', [])  # Expecting a list of questions in the JSON body

    # Load the model and tokenizer
    model_path = 'ai/model/fine_tuned_bert_model.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the label encoder to decode numerical labels to category names
    label_encoder_path = 'ai/model/label_encoder.pkl'
    with open(label_encoder_path, 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    # Process and classify questions
    results = []
    model.eval()  # Set the model to evaluation mode
    device = torch.device("mps" if torch.has_mps else "cpu")
    with torch.no_grad():  # Disable gradient computation for inference
        for question in questions:
            inputs = tokenizer.encode_plus(
                question,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=512
            ).to(device)
            output = model(**inputs)
            prediction = torch.argmax(output.logits, dim=-1).item()  # Extract the class with the highest probability

            # Map the numerical label to the actual category
            category = label_encoder.inverse_transform([prediction])[0]

            results.append({'question': question, 'classification': category})

    return jsonify(results)


@app.route('/feedback', methods=['POST'])
@token_required
def submit_feedback(current_user):
    data = request.get_json()
    content = data.get('content')
    if not content:
        return jsonify({'message': 'No feedback provided'}), 400

    feedback = Feedback(content=content, user_id=current_user.id)
    db.session.add(feedback)
    db.session.commit()

    return jsonify({'message': 'Feedback submitted successfully'}), 201


# Update the endpoint to include user email of the feedback sender
@app.route('/feedback', methods=['GET'])
@token_required
def view_feedback(current_user):
    feedbacks = Feedback.query.filter_by(user_id=current_user.id).order_by(Feedback.creation_date.desc()).all()
    feedback_list = [{
        'id': feedback.id,
        'content': feedback.content,
        'submitted_on': feedback.creation_date.strftime('%Y-%m-%d %H:%M:%S'),
        'user_email': current_user.email  # Include user's email in the response
    } for feedback in feedbacks]

    return jsonify(feedback_list), 200


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
