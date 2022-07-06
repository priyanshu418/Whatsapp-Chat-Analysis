import pickle

# loading the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

loss,acc = model.evaluate(X_test,Y_test)
print("loss : ", loss)
print("accuracy : ", acc)

preds = model.predict(X_test)

# test box in stremlit app
sentence = st.text_input('Input your sentence here:')
if sentence:
    st.write(my_model.predict(sentence))