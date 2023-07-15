import streamlit as st

def blank_page():
  col1, col2, col3 = st.columns(3)
  
  with col1:
    st.subheader("Examples")
    st.info('"What is a glide form used for?"')
    st.info('"Show an example of populating a form field"')
    st.info('"How can I start a work flow using client script?"')
  
  with col2:
    st.subheader("Capabilities")
    st.info('Remembers what was said in a conversation')
    st.info('Allows user to provide follow-up corrections')
    st.info('Trained on ServiceNow documentation')
  
  with col3:
    st.subheader("Limitations")
    st.info('May occasionally generate incorrect information')
    st.info('Code markup is not formatted in this interface')
    st.info('Limited knowledge of world outside ServiceNow')

  st.write(f':page_facing_up: **{st.session_state.selected_option}**')
  st.caption(st.session_state.selected_desc)
