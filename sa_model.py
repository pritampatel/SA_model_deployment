import tensorflow as tf
import tensorflow_datasets as tfds
import streamlit as st
from PIL import Image



#model=tf.keras.models.load_model('drive/MyDrive/model_and_token/sentiment_analysis.hdf5')

st.title("Sentiment Analysis Classifier")
img=Image.open('feat.jpg')
st.image(img,use_column_width=True)


def load_model_tokenizer():
    text_encoder=tfds.deprecated.text.TokenTextEncoder.load_from_file('sa_encoder.vocab')
    model=tf.keras.models.load_model('sentiment_analysis.hdf5')
    return model,text_encoder

st.cache(allow_output_mutation=True)
model,text_encoder=load_model_tokenizer()

st.write("""Model and Encoded imported and cached successfully""")


padding_size=1000
def pad_to_size(vec,size):
    zeros= [0]*(size-len(vec))
    vec.extend(zeros)
    return vec

def predict_fn(pred_text,size=1000):
    text_encoded=text_encoder.encode(pred_text)
    text_encoded= pad_to_size(text_encoded,size=1000)
    encoded_text= tf.cast(text_encoded,tf.int64)
    predictions=model.predict(tf.expand_dims(encoded_text,0))
    sentiment=' Positive' if float(" ".join(map(str,predictions[0])))>0 else ' Negative'
    return sentiment

sentence = st.text_input('Input your sentence here:') 

if sentence:
    st.write('Sentiment of the sentence is:',predict_fn(sentence))
    st.header('**It works great!**')
