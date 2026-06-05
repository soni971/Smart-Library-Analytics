import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Library Analytics",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Background */

.stApp{
    background-image:
    linear-gradient(
        rgba(0,0,0,0.80),
        rgba(0,0,0,0.80)
    ),

    url("https://images.unsplash.com/photo-1521587760476-6c12a4b040da");

    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

/* Hide Streamlit */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Hero Section */

.hero{

    padding:50px;

    border-radius:35px;

    background:
    rgba(255,255,255,0.08);

    backdrop-filter:blur(15px);

    border:
    1px solid rgba(255,255,255,0.1);

    text-align:center;

    box-shadow:
    0px 0px 40px rgba(0,0,0,0.5);
}

/* Main Title */

.main-title{

    font-size:70px;

    font-weight:900;

    background:
    linear-gradient(
        90deg,
        #00E5FF,
        #00FF99,
        #FFD700
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

/* Subtitle */

.sub-title{

    color:white;

    font-size:22px;

    margin-top:15px;
}

/* Cards */

.card{

    background:
    rgba(255,255,255,0.08);

    padding:25px;

    border-radius:25px;

    backdrop-filter:blur(12px);

    border:
    1px solid rgba(255,255,255,0.1);

    text-align:center;
}

/* Text */

h1,h2,h3,h4,h5,h6,p,label,div{
    color:white !important;
}

/* Buttons */

.stButton>button{

    width:100%;

    height:60px;

    border:none;

    border-radius:15px;

    background:
    linear-gradient(
        to right,
        #00E5FF,
        #00FF99
    );

    color:black;

    font-size:20px;

    font-weight:bold;
}

/* Input Fields */

.stNumberInput{
    background:rgba(255,255,255,0.05);
}

/* Metrics */

.metric-card{

    background:
    rgba(255,255,255,0.08);

    padding:20px;

    border-radius:20px;

    text-align:center;

    border:
    1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""

<div class="hero">

<div class="main-title">
📚 Smart Library Analytics
</div>

<div class="sub-title">

AI Powered Frequent User Prediction System

<br><br>

Predict whether a student is a frequent
library visitor using Machine Learning.

</div>

</div>

""", unsafe_allow_html=True)

st.write("")

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("library_data.csv")

df = load_data()

# ==========================================
# MACHINE LEARNING
# ==========================================

X = df[
    [
        "StudentAge",
        "BooksIssued",
        "LateReturns",
        "MembershipYears"
    ]
]

y = df["FrequentUser"]

le = LabelEncoder()

y_encoded = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(
    x_train,
    y_train
)

pred = model.predict(x_test)

accuracy = accuracy_score(
    y_test,
    pred
)

# ==========================================
# DASHBOARD
# ==========================================

st.subheader("📊 Library Intelligence Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "👨‍🎓 Students",
        len(df)
    )

with c2:
    st.metric(
        "📚 Avg Books",
        round(df["BooksIssued"].mean(), 1)
    )

with c3:
    st.metric(
        "⏳ Avg Membership",
        round(df["MembershipYears"].mean(), 1)
    )

with c4:
    st.metric(
        "🎯 Accuracy",
        f"{accuracy*100:.2f}%"
    )

st.write("")

# ==========================================
# ACCURACY CARD
# ==========================================

st.markdown(f"""
<div class="card">

<h2>🤖 AI Model Performance</h2>

<h1>{accuracy*100:.2f}%</h1>

<p>
Decision Tree Classifier trained successfully
for frequent user prediction.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# PREDICTION SECTION
# ==========================================

st.subheader("🔮 Predict Student Library Activity")

col1, col2 = st.columns(2)

with col1:

    student_age = st.number_input(
        "👨‍🎓 Student Age",
        min_value=5,
        max_value=100,
        value=18
    )

    books_issued = st.number_input(
        "📚 Books Issued",
        min_value=0,
        value=5
    )

with col2:

    late_returns = st.number_input(
        "⏰ Late Returns",
        min_value=0,
        value=1
    )

    membership_years = st.number_input(
        "📅 Membership Years",
        min_value=0,
        value=2
    )

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🚀 Predict User Type"):

    new_data = [[
        student_age,
        books_issued,
        late_returns,
        membership_years
    ]]

    prediction = model.predict(
        new_data
    )

    st.write("")

    if prediction[0] == 0:

        st.success(
            "✅ Prediction Result: FREQUENT USER"
        )

        st.balloons()

        st.info(
            "This student actively uses library resources and visits frequently."
        )

    else:

        st.error(
            "❌ Prediction Result: NOT A FREQUENT USER"
        )

        st.warning(
            "This student may require encouragement to use library resources more often."
        )

# ==========================================
# FOOTER
# ==========================================

st.write("")
st.write("")

st.markdown("""

<hr>

<center>

<h3 style="color:white;">
📚 Smart Library Analytics Platform
</h3>

<p style="color:lightgray;">
Built with Streamlit, Pandas, Scikit-Learn and Machine Learning
</p>

<p style="color:gray;">
Next Generation Library Intelligence System
</p>

</center>

""", unsafe_allow_html=True)