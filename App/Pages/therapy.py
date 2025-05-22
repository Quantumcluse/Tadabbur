import streamlit as st

# Placeholder for loading the model
# import pickle
# with open("therapy_model.pkl", "rb") as f:
#     model = pickle.load(f)

# Dummy therapy function
def get_therapeutic_verse(emotion):
    # Replace this with actual model inference
    dummy_data = {
        "Sadness": {
            "verse": "Surah Al-Inshirah (94:5-6)",
            "text": "For indeed, with hardship [will be] ease. Indeed, with hardship [will be] ease."
        },
        "Anxiety": {
            "verse": "Surah Ar-Ra'd (13:28)",
            "text": "Verily, in the remembrance of Allah do hearts find rest."
        },
        "Anger": {
            "verse": "Surah Al-Imran (3:134)",
            "text": "Who restrain anger and pardon the people – and Allah loves the doers of good."
        },
        "Loneliness": {
            "verse": "Surah At-Tawbah (9:40)",
            "text": "Do not grieve; indeed Allah is with us."
        }
    }
    return dummy_data.get(emotion, None)

# Main function for this page

st.title("Quran Therapy")
st.write("Select an emotion to receive comforting guidance from the Quran.")

    # Emotion dropdown
emotion = st.selectbox(
        "Choose an emotion",
        options=["", "Sadness", "Anxiety", "Anger", "Loneliness"],
        index=0,
        format_func=lambda x: "Select..." if x == "" else x,
        label_visibility="collapsed"
    )

if emotion:
    st.markdown(f"### 💬 Therapy for *{emotion}*")

    result = get_therapeutic_verse(emotion)

    if result:
        st.subheader(result["verse"])
        st.write(result["text"])
    else:
        st.warning("No therapy content available for the selected emotion.")


