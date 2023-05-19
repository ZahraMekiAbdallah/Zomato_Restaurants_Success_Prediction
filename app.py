import pickle as pkl
import streamlit as st
import pandas as pd
import joblib

cuisines_lst = joblib.load('cuisines.sav')
location_lst = joblib.load('location.sav')
city_lst = joblib.load('city.sav')
rest_type_lst = joblib.load('rest_type.sav')

# Take inputs from user
st.image('img.jpg')
cost = st.slider("Cost", 40, 6000)
online_order = st.checkbox('Allow Online Orders ?')
book_table = st.checkbox('Allow Table Booking ?')
has_online_menu = st.checkbox('Has Online Menu ?')
list_type = st.selectbox('Type', ['Delivery','Dine-out','Desserts','Cafes','Drinks & nightlife','Buffet', 'Pubs and bars'])
location = st.selectbox('Location', location_lst)
city = st.selectbox('City', city_lst)
rest_type = st.selectbox('Restuarant Type', rest_type_lst)
cuisines = st.selectbox('Cuisines', cuisines_lst)

# Get num_cuisines
num_cuisines = len(cuisines.split(', '))

num_cols = ['cost', 'num_cuisines']
cat_cols = ['location', 'rest_type', 'cuisines', 'city', 'type']
bool_cols = ['online_order', 'book_table', 'has_online_menu']

# Convert inputs to DataFrame
df_new = pd.DataFrame({'cost': [cost], 'num_cuisines': [num_cuisines], 
                'location': [location], 'rest_type':[rest_type],'cuisines': [cuisines], 'city': [city], 'type': [list_type],
                'online_order': [online_order],'book_table': [book_table],'has_online_menu': [has_online_menu]
		     })

# Load the transformer
transformer = pkl.load(open('trans.pkl', 'rb'))
#transformer = joblib.load(open('trans.joblib', 'rb'))
# Apply the transformer on the inputs
X = transformer.transform(df_new)

# Load the model
model = pkl.load(open('f_clf.pkl', 'rb'))
#model = joblib.load(open('f_clf.joblib', 'rb'))
# Predict the output
predict = model.predict(X)
prop = model.predict_proba(X)[0][1] * 100
print(model.predict_proba(X))
st.markdown(f'## Success With Probability : {round(prop, 2)} %')

