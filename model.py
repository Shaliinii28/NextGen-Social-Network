from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func


db=SQLAlchemy()

# Association table for followers
followers = db.Table('followers',
    db.Column('follower_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('following_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100),nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    profile_picture = db.Column(db.String(255))
    bio=db.Column(db.String(255))
    posts =  db.relationship("Post", backref="user",cascade="all, delete")
    
     # Many-to-many relationship: users following this user
    followers = db.relationship('User', secondary=followers,
        primaryjoin=(followers.c.following_id == id),
        secondaryjoin=(followers.c.follower_id == id),
        backref=db.backref('following', lazy='dynamic'), lazy='dynamic')
    
    
    def follow(self, user):
        if not self.is_following(user):
            self.following.add(user)

    def unfollow(self, user):
        if self.is_following(user):
            self.following.remove(user)

    def is_following(self, user):
        return self.following.filter_by(id=user.id).first() is not None
    
    def followers_count(self):
        return self.followers.count()
    
    def following_count(self):
        return self.following.count()
    
    def get_news_feed_posts(self):
        followed_users_posts = Post.query.join(
            followers, (followers.c.following_id == Post.user_id)
        ).filter(
            followers.c.follower_id == self.id
        ).order_by(
            Post.timestamp.desc()
        ).all()
        return followed_users_posts
    # Method to get list of followers
    def get_followers(self):
        # Return a list of user objects representing the followers of this user
        return self.followers.all()

    # Method to get list of following users
    def get_following(self):
        # Return a list of user objects representing the users this user is following
        return self.following.all()
   

    

class Post(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    content=db.Column(db.String(500)) 
    image = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime(timezone=True),default=func.now())
    user_id = db.Column(db.Integer,db.ForeignKey('user.id'),nullable = False)
    tag= db.Column(db.String(255))
    caption=db.Column(db.String(255))
    sentiment = db.Column(db.Integer, nullable=False)
    toxicity = db.Column(db.Integer, nullable=False)

    
    
    

class User_Search(db.Model):
    rowid=db.Column(db.Integer, primary_key=True)
    name= db.Column(db.String(100))
    username = db.Column(db.String(50))
    
    
    
    


   