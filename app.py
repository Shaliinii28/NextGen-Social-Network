from flask import Flask, jsonify, redirect,render_template,request,session,flash,url_for
from model import Post, db, User, User_Search
from flask_bcrypt import Bcrypt
from sqlalchemy import desc
from werkzeug.utils import secure_filename
import os
import pickle
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_preprocessing
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import torch
from sqlalchemy import func

# Initialize BlipProcessor and BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")





app=Flask(__name__)
bcrypt = Bcrypt(app)

upload_folder=os.path.join('static','uploads')
image_folder=os.path.join('static','image_post')




app.config['UPLOAD']=upload_folder
app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///mydb.sqlite3" 
app.config["SECRET_KEY"]= '#123456654321yrtiam'
app.config["IMAGE_POST"]=image_folder


db.init_app(app)

app.app_context().push()
db.create_all()


#################################### MACHINE LEARNING MODELS- SENTIMENT ANALYSIS  #####################################
# Loading the pre-trained model and vectorizer
def load_models():
    with open('vectoriser.sav', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('trained_model.sav', 'rb') as file:
        model = pickle.load(file)
    return vectorizer, model

vectorizer, model = load_models()



#################################### MACHINE LEARNING MODELS - HATE SPEECH DETECTION  #####################################

# Loading the pre-trained model and vectorizer
def load_models():
    with open('hate_vector', 'rb') as file:
        hate_vector = pickle.load(file)
    with open('hate_model', 'rb') as file:
        hate_model = pickle.load(file)
    return hate_vector, hate_model

hate_vector,hate_model = load_models()



# Function to preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuations, special characters, and lemmatize words
    text_lem = ''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ', text))])
    # Tokenize the text
    word_list = word_tokenize(text_lem)
    # Remove stopwords
    useful_words = [w for w in word_list if w not in stopwords.words("english")]
    # Join the words back into a sentence
    preprocessed_text = ' '.join(useful_words)
    return preprocessed_text







#################################### PREPROCESSING  #############################################

stop_words = set(stopwords.words('english'))
port_stem=PorterStemmer()

def preprocessing(content):
    # Removing non-alphabetic characters and replace with spaces
    clean_content = re.sub('[^a-zA-Z]', ' ', content)

    # Converting to lowercase
    clean_content = clean_content.lower()

    # Tokenizing the text into words
    clean_content = clean_content.split()

    # Stemming each word and remove stopwords
    clean_content = [port_stem.stem(word) for word in clean_content if word not in stop_words]

    # Joining the stemmed words back into a single string
    clean_content = ' '.join(clean_content)

    return clean_content





#############################################   TRENDING   ######################################
def get_tag_counts():
    tag_counts = Counter()
    for post in Post.query.all():
        if post.tag:
            tag_counts[post.tag] += 1
    return tag_counts

def sort_tags_by_count(tag_counts):
    return sorted(tag_counts.items(), key=lambda item: item[1])
  


@app.route("/<int:id>/trending")
def trending(id):
    this_user=User.query.get(id)
    tag_counts = sort_tags_by_count(get_tag_counts())
    return  render_template("trending.html",this_user=this_user,sorted_tags=tag_counts)



#######################################  TAG   ###############################################





@app.route("/tag/<tag_name>")
def tag_detail(tag_name):
    this_user = User.query.get(session['user_id'])
    tag_posts = Post.query.filter(Post.tag == tag_name).all()
    for post in tag_posts:
        preprocessed_content = preprocessing(post.content)
        textdata = vectorizer.transform([preprocessed_content])
        sentiment = model.predict(textdata)
        print(sentiment)
        post.sentiment = int(sentiment[0])
        db.session.commit()
    return render_template("tag.html", tag_name=tag_name, tag_posts=tag_posts,this_user=this_user)



@app.route("/tag2/<tag_name>")
def tag_detail2(tag_name):
    this_user = User.query.get(session['user_id'])
    tag_posts = Post.query.filter(Post.tag == tag_name).all()
    for post in tag_posts:
        preprocessed_content = preprocessing(post.content)
        textdata = vectorizer.transform([preprocessed_content])
        sentiment = model.predict(textdata)
        print(sentiment)
        post.sentiment = int(sentiment[0])
        db.session.commit()
        return render_template("tag2.html", tag_name=tag_name, tag_posts=tag_posts,this_user=this_user)


#################################### HANDLING POST  #####################################
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/<int:id>/upload", methods=["GET", "POST"])
def upload(id):
    this_user = User.query.get(id)
    image_path = this_user.profile_picture
    image = None

    if request.method == "POST":
        content = request.form.get("text-post")
        
        # Preprocess the content to check for toxicity
        preprocessed_content = preprocessing(content)
        
        # Vectorize the preprocessed content using the TF-IDF vectorizer
        content_vectorized = hate_vector.transform([preprocessed_content])
        
        # Predict toxicity using the trained logistic regression model
        toxicity_prediction = hate_model.predict(content_vectorized)
        
        print("Toxicity Prediction:", toxicity_prediction)

        if toxicity_prediction == 1:
            return render_template("toxic.html", this_user=this_user)
        
        if toxicity_prediction == 0:
            file = request.files.get("fileInput")
            
            # Check if the text area is empty
            if not content.strip() and (file is None or file.filename == ''):
                return render_template("post.html", this_user=this_user, image_path=image_path)
            
            tag = request.form.get("tag")
            
            if file:
                filename = secure_filename(file.filename)
                image = os.path.join(app.config["IMAGE_POST"], filename)
                file.save(image)
                
                # Open the saved image file
                raw_image = Image.open(image).convert('RGB')
                
                # Generate a caption only if the text area is empty
                if not content.strip():
                    inputs = processor(raw_image, return_tensors="pt")
                    outputs = caption_model.generate(**inputs)
                    caption_unconditional = processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Use the generated caption as content
                    content = caption_unconditional
            
            # Create a new Post object
            this_post = Post(content=content, tag=tag, user_id=this_user.id, image=image)
            
            # Add the new Post object to the database session
            db.session.add(this_post)
            
            # Commit the changes to the database
            db.session.commit()
            
            return render_template("uploaded.html", this_post=this_post, this_user=this_user, image_path=image_path, image=image_path)

    return render_template("post.html", this_user=this_user, image_path=image_path)






######################################### SEARCH FOR USER ###############################

@app.route('/<int:id>/search')
def search(id):
    this_user=User.query.get(id)
    query = request.args.get('query')
    if query:
        # Perform search logic and retrieve results
        results = User.query.filter(
            User.username.ilike("%" + query + "%") |
            User.name.ilike("%" + query + "%") ).all()
    else:
        results = []  # Empty list if no query provided
    return render_template('user_search_result.html', results=results,this_user=this_user,query=query)






#########################################   FOLLOW USER   ######################################


@app.route('/<int:id>/follows/<int:following_id>', methods=['POST'])
def follow(id, following_id):
    this_user = User.query.get(id)
    query = request.form.get('query')
    if not this_user or this_user.id == following_id:
        flash('Invalid follow request', 'error')
        return redirect(url_for('search', id=id, query=query))

    following_user = User.query.get(following_id)
    if not following_user:
        flash('User not found', 'error')
        return redirect(url_for('search', id=id, query=query))

    if this_user.is_following(following_user):
        flash('Already following this user', 'info')
        return redirect(url_for('search', id=id, query=query))

    this_user.follow(following_user)
    db.session.commit()

    flash('Following successful', 'success')
    return redirect(url_for('search', id=id, query=query))

    
    



#########################################   UNFOLLOW USER   ###############################



@app.route('/<int:id>/unfollows/<int:following_id>', methods=['POST'])
def unfollow(id,following_id):
    this_user = User.query.get(id)
    this_user = User.query.get(id)
    if not this_user:
        flash('User not found', 'error')
        return redirect(url_for('search', id=id, query=request.form.get('query')))

    following_user = User.query.get(following_id)
    if not following_user:
        flash('User to unfollow not found', 'error')
        return redirect(url_for('search', id=id, query=request.form.get('query')))

    if not this_user.is_following(following_user):
        flash('You are not following this user', 'error')
        return redirect(url_for('search', id=id, query=request.form.get('query')))

    this_user.unfollow(following_user)
    db.session.commit()

    flash('Unfollow successful', 'success')
    return redirect(url_for('search', id=id, query=request.form.get('query')))
    





######################################### USER PROFILE ###############################

# 1] View user profile
@app.route("/<int:id>/user",methods=["GET","POST"])
def user(id):
    this_user=User.query.get(id)
    image_path=this_user.profile_picture
    
    user_posts = Post.query.filter_by(user_id=this_user.id).order_by(desc(Post.timestamp)).all()
    post_count = db.session.query(func.count(Post.id)).filter_by(user_id=this_user.id).scalar()
    return render_template("profile.html",this_user=this_user,image_path=image_path,user_posts=user_posts)




# 2] User Feed
@app.route("/<int:id>/feed",methods=["GET","POST"])
def feed(id):
    this_user=User.query.get(id)
    posts = this_user.get_news_feed_posts()
    return render_template("feed.html",this_user=this_user,posts=posts)




# 3] Edit user profile
@app.route("/<int:id>/profile/edit",methods=["GET","POST"])
def edit_profile(id):
    this_user=User.query.get(id)
    if request.method=="POST":
        uptd_name=request.form.get("name")
        this_user.name=uptd_name
        
        uptd_bio=request.form.get("bio")
        this_user.bio=uptd_bio
        
        uptd_password=request.form.get("password")
        this_user.password= bcrypt.generate_password_hash(uptd_password).decode('utf-8')
        
        file = request.files.get('profile_picture')        #this_user.profile_picture=uptd_picture
        if file:
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD'],filename))
            profile_picture=os.path.join(app.config['UPLOAD'],filename)
            this_user.profile_picture=profile_picture

        
        db.session.commit()
        return render_template("profile.html",this_user=this_user)
    return render_template("edit_profile.html",this_user=this_user)

from face_detection import FaceVerification

@app.route("/<int:id>/profile/verify",methods=["GET","POST"])
def face_verify(id):
    this_user=User.query.get(id)

    pic=this_user.profile_picture
    a,b=FaceVerification([pic])
    if a==True:
        print("Verified")
        this_user.verified='1'
    else:
        print("Not verified")
    db.session.commit()
    return render_template("profile.html",this_user=this_user)




# 4] Delete Profile
@app.route("/<int:id>/profile/delete",methods=["GET","POST"])
def delete_profile(id):
    this_user=User.query.get(id)
    if request.method=="POST":
        action=request.form.get("action")
        print("Action received:", action)
        if action=="delete":
                db.session.delete(this_user)
                db.session.commit()
                return redirect('/login')
        if action=="cancel":
                return render_template("profile.html",this_user=this_user)
    return render_template("delete_profile.html",this_user=this_user)
    






##############################################   LOGIN    #######################################################

@app.route("/login",methods=["GET","POST"])
def login():
    if request.method=="POST":
        username = request.form.get('username')
        password = request.form.get('password')
        #validation
        already_user = User.query.filter_by(username=username).first()
        if already_user and  bcrypt.check_password_hash(already_user.password, password):
            # Password is correct, set up the user session
            session['user_id'] = already_user.id
            return redirect(f"/{session['user_id']}/user")
        else:
            # Invalid credentials
            return render_template("login.html", invalid_credentials=True)
    
    return render_template("login.html")



@app.route("/<int:id>/logout")
def logout(id):
    this_user=User.query.get(id)
    # Remove user ID from session
    session.pop(this_user, None)
    # Redirect to login page
    return redirect("/login")








######################################    REGISTERATION    ###############################

@app.route("/register",methods=["GET","POST"])
def register():
    if request.method=="POST":
        name=request.form.get('name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        file = request.files.get('profile_picture')
        bio=request.form.get('bio')
        
        
        if not username or not password or not email:
            return "Please fill in all required fields."

        
        '''Validation for username characters'''
        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists! Please choose a different one."
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "This email is already registered. Please choose a different email"
        if not email or not re.match(r"^[^@]+@[^@]+\.[a-zA-Z]{2,}$", email):
            return "Please enter a valid email address."
        
        if file:
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD'],filename))
            profile_picture=os.path.join(app.config['UPLOAD'],filename)
        
        # Create a new User object
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(name=name,username=username, email=email, password=hashed_password, profile_picture=profile_picture, bio=bio)
        
        
        # Add the new User object to the database session
        db.session.add(new_user)
        
        
        # Commit the changes to the database
        db.session.commit()

        
        return redirect("/login")
    
    return render_template("register.html")



################################ FOLLOWERS & FOLLOWING DISPLAY #############################
@app.route('/<int:user_id>/profile/followers', methods=['GET'])
def display_followers(user_id):
    # Get the current user by ID
    this_user = User.query.get(user_id)
    
    if not this_user:
        flash('User not found', 'error')
        return redirect(url_for('home'))

    # Get the list of followers
    followers = this_user.get_followers()

    # Render the followers page
    return render_template('followers.html', this_user=this_user, followers=followers)

@app.route('/<int:user_id>/profile/following', methods=['GET'])
def display_following(user_id):
    # Get the current user by ID
    this_user = User.query.get(user_id)
    
    if not this_user:
        flash('User not found', 'error')
        return redirect(url_for('home'))

    # Get the list of following users
    following = this_user.get_following()

    # Render the following page
    return render_template('following.html', this_user=this_user, following=following)

##########################SUMMARY ####################################

from transformers import pipeline
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize explanation pipeline
explainer = pipeline("summarization", model="Mr-Vicky-01/Bart-Finetuned-conversational-summarization")

# Function to explain all post contents for a user
def explain_posts(user_id):
    this_user = User.query.get(user_id)
    try:
        # Fetch all posts for the user
        posts = this_user.get_news_feed_posts()
        
        # Concatenate the content of all posts, filtering out None values
        all_content = " ".join(post.content for post in posts if post.content is not None)
        
        # Log the length of the content for debugging
        logging.debug(f"Total content length: {len(all_content)} characters")
        
        # Check if there is enough content
        if len(all_content) == 0:
            logging.warning("No content available for explanation.")
            return "No content available for explanation."
        
        # Generate the explanation
        explanation = explainer(all_content)
        
        # Log the generated explanation
        logging.debug(f"Generated explanation: {explanation}")
        
        # Return the explanation text
        return explanation
    
    except Exception as e:
        logging.error(f"Error while explaining posts: {e}")
        return "An error occurred while explaining posts."



@app.route("/<int:id>/summary")
def display_summary(id):
    # Fetch the user
    this_user = User.query.get(id)
    
    if not this_user:
        flash('User not found', 'error')
        return redirect(url_for('home'))
    
    # Get the summary of all posts by the user
    summary_result = explain_posts(this_user.id)
    
    # Extract the summary text from the returned result
    summary_text = summary_result[0].get("summary_text", "")
    
    # Render the summary template
    return render_template("summary.html", this_user=this_user, summary=summary_text)


#################################### RUNNING THE APP  #####################################

if __name__=="__main__":
    app.run(debug=True)
    
    
