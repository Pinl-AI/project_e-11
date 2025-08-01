import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import random
import tensorflow as tf

def compute_k(temp, Ea, A_factor):
    R = 8.314
    Ea_J = Ea * 1000  # Convert Ea from kJ/mol to J/mol
    k = A_factor * np.exp(-Ea_J / (R * temp))
    return k

def zero(t, y, k):
    A, B, C = y
    dA_dt = -k
    dB_dt = 0
    dC_dt = k
    return [dA_dt, dB_dt, dC_dt]

def first(t, y, k):
    A, B, C = y
    dA_dt = -k * A
    dB_dt = 0
    dC_dt = +k * A
    return [dA_dt, dB_dt, dC_dt]

def decay_first(t, y, k):
    A, B, C = y
    dA_dt = -k * A
    dB_dt = 0
    dC_dt = 0
    return [dA_dt, dB_dt, dC_dt]

def reversible_first(t, y, k, k_1):
    A, B, C = y
    dA_dt = -k * A + k_1 * C
    dB_dt = 0
    dC_dt = k * A - k_1 * C
    return [dA_dt, dB_dt, dC_dt]

def second1(t, y, k):
    A, B, C = y
    dA_dt = -k * A * B
    dB_dt = -k * A * B
    dC_dt = +k * A * B
    return [dA_dt, dB_dt, dC_dt]

def second2(t, y, k):
    A, B, C = y
    dA_dt = -2 * k * A**2
    dB_dt = 0
    dC_dt = +k * A**2
    return [dA_dt, dB_dt, dC_dt]

def reversible_second1(t, y, k, k_1):
    A, B, C = y
    dA_dt = -k * A * B + k_1 * C
    dB_dt = -k * A * B + k_1 * C
    dC_dt = +k * A * B - k_1 * C
    return [dA_dt, dB_dt, dC_dt]

def reversible_second2(t, y, k, k_1):
    A, B, C = y
    dA_dt = -2 * k * A**2 + 2 * k_1 * C
    dB_dt = 0
    dC_dt = +k * A**2 - k_1 * C
    return [dA_dt, dB_dt, dC_dt]

def third1(t, y, k):
    A, B, C = y
    dA_dt = -3 * k * A**3
    dB_dt = 0
    dC_dt = +k * A**3
    return [dA_dt, dB_dt, dC_dt]

def third2(t, y, k):
    A, B, C = y
    dA_dt = -2 * k * A**2 * B
    dB_dt = -1 * k * A**2 * B
    dC_dt = +k * A**2 * B
    return [dA_dt, dB_dt, dC_dt]

def reversible_third1(t, y, k, k_1):
    A, B, C = y
    dA_dt = -3 * k * A**3 + 3 * k_1 * C
    dB_dt = 0
    dC_dt = +k * A**3 - k_1 * C
    return [dA_dt, dB_dt, dC_dt]

def reversible_third2(t, y, k, k_1):
    A, B, C = y
    dA_dt = -2 * k * A**2 * B + 2 * k_1 * C
    dB_dt = -1 * k * A**2 * B + 1 * k_1 * C
    dC_dt = +k * A**2 * B - k_1 * C
    return [dA_dt, dB_dt, dC_dt]


def ode1(A0, B0, C0, temp, Ea, A_factor):
  y0 = [A0, B0, C0]
  k = compute_k(temp, Ea, A_factor)
  k_1 = k * random.uniform(0.5, 0.9)

  t_span = (0, 8)  # From time 0 to 10 seconds
  t_eval = np.linspace(0, 8, 11)  # 11 points where you want the solution

  num = random.randint(0, 11)   # For choosing between first or decay if not reversible

  match num:
    case 0:
      func_name = zero
      is_reversible = 0
      order = 'zero'
    case 1:
      func_name = first
      is_reversible = 0
      order = 'first'
    case 2:
      func_name = decay_first
      is_reversible = 0
      order = 'first'
    case 3:
      func_name = reversible_first
      is_reversible = 1
      order = 'first'
    case 4:
      func_name = second1
      is_reversible = 0
      order = 'second'
    case 5:
      func_name = second2
      is_reversible = 0
      order = 'second'
    case 6:
      func_name = reversible_second1
      is_reversible = 1
      order = 'second'
    case 7:
      func_name = reversible_second2
      is_reversible = 1
      order = 'second'
    case 8:
      func_name = third1
      is_reversible = 0
      order = 'third'
    case 9:
      func_name = third2
      is_reversible = 0
      order = 'third'
    case 10:
      func_name = reversible_third1
      is_reversible = 1
      order = 'third'
    case 11:
      func_name = reversible_third2
      is_reversible = 1
      order = 'third'


  if is_reversible == 1:
    solution = solve_ivp(
      func_name,
      t_span,
      y0,
      args=(k, k_1),
      t_eval=t_eval
    )
  elif is_reversible == 0:
    solution = solve_ivp(
      func_name,
      t_span,
      y0,
      args=(k,),
      t_eval=t_eval
      )


  return solution.t, solution.y[0], solution.y[1], solution.y[2], k, k_1, is_reversible, order

results = []

counter = 0
while counter < 6000:
    counter += 1

    A0 = round(random.uniform(1.0, 10.0), 2)
    B0 = round(random.uniform(0.0, 5.0), 2)
    C0 = round(random.uniform(0.0, 5.0), 2)
    temp = random.randint(270, 280)
    pH = round(random.uniform(1.0, 14.0), 2)
    Ea = random.randint(90, 100)
    A_factor = round(random.uniform(2e16, 5e17), 2)
    pressure = round(random.uniform(0.5, 5.0), 2)
    weight = round(random.uniform(20, 200), 1)
    structure = random.choice(['Linear', 'Ring', 'Branched', 'Unknown'])
    catalyst = random.choice(['None', 'Enzyme', 'Acid', 'Base'])
    time, A, B, C, k, k_1, is_reversible, order  = ode1(A0, B0, C0, temp, Ea, A_factor)

    row = {
        'order' : order,
        'temp': temp,
        'pH': pH,
        'Ea': Ea,
        'A_factor': A_factor,
        'pressure': pressure,
        'log_pressure' : np.log(pressure),
        'weight': weight,
        'structure': structure,
        'catalyst': catalyst,
        'is_reversible': is_reversible,
        'k' : k,
        'k_1' : k_1,
        'A0': A[0], 'A1': A[1], 'A2': A[2], 'A3': A[3], 'A4': A[4],
        'A5': A[5], 'A6': A[6], 'A7': A[7], 'A8': A[8], 'A9': A[9], 'A10': A[10],
        'B0': B[0], 'B1': B[1], 'B2': B[2], 'B3': B[3], 'B4': B[4],
        'B5': B[5], 'B6': B[6], 'B7': B[7], 'B8': B[8], 'B9': B[9], 'B10': B[10],
        'C0': C[0], 'C1': C[1], 'C2': C[2], 'C3': C[3], 'C4': C[4],
        'C5': C[5], 'C6': C[6], 'C7': C[7], 'C8': C[8], 'C9': C[9], 'C10': C[10]
    }
    results.append(row)

df = pd.DataFrame(results)
df_original = df.copy()
# display(df)

structure_map = {'Linear': 0, 'Ring': 1, 'Branched': 2, 'Unknown': 3}
catalyst_map = {'None': 0, 'Enzyme': 1, 'Acid': 2, 'Base': 3}
order_map = {'zero': 0, 'first': 1, 'second': 2, 'third' : 3}
df['structure'] = df['structure'].map(structure_map)
df['catalyst'] = df['catalyst'].map(catalyst_map)
df['order'] = df['order'].map(order_map)
# display(df)


csv_columns = ['temp', 'pH', 'Ea', 'A_factor', 'pressure', 'log_pressure', 'weight', 'structure', 'catalyst', 'is_reversible', 'k', 'k_1']
classes = ['First_Order','Second_Order','Third_Order']

train_path = './chem_data_train.csv'
test_path = './chem_data_train.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# display(train.head())

if 'order' in train.columns:
    train_y = train.pop('order')
if 'order' in test.columns:
    test_y = test.pop('order')

# Fill missing values in the 'catalyst' column
train['catalyst'] = train['catalyst'].fillna('None') #NaN values arenot accepted by classifier thats why convert every Nan values to none
test['catalyst'] = test['catalyst'].fillna('None')


# display(train.head()) #the species column is now gone

# Define categorical and numerical feature columns
CATEGORICAL_COLUMNS = ['structure', 'catalyst'] #columns that have strings
NUMERIC_COLUMNS = ['temp', 'pH', 'Ea', 'A_factor', 'pressure', 'log_pressure', 'weight',
                   'is_reversible', 'k', 'k_1', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                   'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
                   'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'] #columns that have numerical values

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = train[feature_name].unique() #Assining each string a numerical uinque value because our dumb ahh model canot understand english
  cat_column = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  indicator_column = tf.feature_column.indicator_column(cat_column) #it creates binary coolumns that will be mapped in to feature columns and it will be steamlined to our DNN model
  feature_columns.append(indicator_column)

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# print(feature_columns)

import logging
tf.get_logger().setLevel(logging.INFO)

#setting up input function

def input_fn(features,labels,training=True,batch_size=500):
  #convert the inputs to a dataset
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)) #this cnonverts the dataset into tensorflow object

  if training:
    dataset = dataset.shuffle(1000).repeat()

  return dataset.batch(batch_size)

from sklearn.preprocessing import StandardScaler

# Normalize the numerical features in the training data
scaler = StandardScaler()
train_normalized = train.copy()
train_normalized[NUMERIC_COLUMNS] = scaler.fit_transform(train[NUMERIC_COLUMNS])

test_normalized = test.copy()
test_normalized[NUMERIC_COLUMNS] = scaler.transform(test[NUMERIC_COLUMNS])

from sklearn.preprocessing import LabelEncoder

# Convert the 'order' labels to numerical values
le = LabelEncoder()
train_y_encoded = le.fit_transform(train_y) #we used sckit label encoder to encode the values

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[50, 40],
    n_classes=4,  # We have 4 classes: zero, first, second, third
    optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001))

classifier.train(
    input_fn=lambda: input_fn(train_normalized, train_y_encoded, training=True),
    steps=300
)

test_y_encoded = le.fit_transform(test_y) #we used sckit label encoder to encode the values better than 1 2 3 4 5 blah blah

classifier.evaluate(input_fn=lambda: input_fn(test_normalized,test_y_encoded,training=False))


def predict_order(inputs):

  try:
    # Create a pandas DataFrame from the input dictionary
    input_df = pd.DataFrame(inputs, index=[0])

    # Normalize the numerical features
    input_df[NUMERIC_COLUMNS] = scaler.transform(input_df[NUMERIC_COLUMNS])

    # Make a prediction
    predictions = classifier.predict(input_fn=lambda: input_fn(input_df, labels=None, training=False))

    # Get the predicted class and probability
    for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]
      # Get the class name from the label encoder
      class_name = le.inverse_transform([class_id])[0]
      print('Order is "{}" ({:.1f}%)'.format(class_name, 100 * probability))
      return class_name
  except Exception as e:
    print(f"An error occurred: {e}")
    return None

def ode2(A0, B0, C0, temp, Ea, A_factor, is_reversible, order):
  y0 = [A0, B0, C0]

  k = compute_k(temp, Ea, A_factor)
  k_1 = k * 0.7

  t_span = (0, 8)
  t_eval = np.linspace(0, 8, 11)

  if order == 'zero':
    solution = solve_ivp(zero, t_span, y0, args=(k,) ,t_eval=t_eval)
  elif is_reversible == 0 and order == 'first':
    solution = solve_ivp(first, t_span, y0, args=(k,) ,t_eval=t_eval)
  elif is_reversible == 1 and order == 'first':
    solution = solve_ivp(reversible_first, t_span, y0, args=(k, k_1) ,t_eval=t_eval)
  elif is_reversible == 0 and order == 'second':
    solution = solve_ivp(second1, t_span, y0, args=(k,) ,t_eval=t_eval)
  elif is_reversible == 1 and order == 'second':
    solution = solve_ivp(reversible_second1, t_span, y0, args=(k, k_1) ,t_eval=t_eval)
  elif is_reversible == 0 and order == 'third':
    solution = solve_ivp(third2, t_span, y0, args=(k,) ,t_eval=t_eval)
  elif is_reversible == 1 and order == 'third':
    solution = solve_ivp(reversible_third2, t_span, y0, args=(k, k_1) ,t_eval=t_eval)

  return solution.t, solution.y[0], solution.y[1], solution.y[2], k, k_1



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming the functions compute_k, ode1, ode2, predict_order, and the classifier, scaler, and le objects are already defined and available in the notebook's global scope from previous cells.

st.set_page_config(layout="wide", page_title="Chemical Reaction Simulator") # Set page layout to wide and add a page title

st.title("🧪 Project E-11")
st.markdown("🧪 Chemical Reaction Order Prediction and Simulation ✨")
st.markdown("Adjust the parameters below to predict the reaction order and visualize the concentration changes over time. 👇")

# Use columns for a better layout of inputs
col1, col2 = st.columns(2)

with col1:
    st.header("⚙️ Reaction Conditions")
    temp = st.slider("Temperature (K) 🌡️", 270.0, 280.0, value=277.0)
    Ea = st.slider("Activation Energy (Ea, kJ/mol) 🔥", 90.0, 100.0, value=93.0)
    A_factor = st.slider("Pre-exponential Factor (A_factor) 📈", 2e16, 5e17, value=4.2e17, format="%e") # Use scientific notation format
    pH = st.slider("pH 🧪", 1.0, 14.0, value=6.5)
    pressure = st.slider("Pressure 🌫️", 0.5, 5.0, value=3.0)
    is_reversible = st.checkbox("Is Reversible? 🔄", value=False)
    structure = st.selectbox("Structure ⚛️", ['Linear', 'Ring', 'Branched', 'Unknown'], index=1)
    catalyst = st.selectbox("Catalyst ✨", ['None', 'Enzyme', 'Acid', 'Base'], index=2)

with col2:
    st.header("📈 Initial Concentrations")
    A0 = st.slider("Initial Concentration of A (A₀)", 0.0, 10.0, value=5.0)
    B0 = st.slider("Initial Concentration of B (B₀)", 0.0, 10.0, value=2.0)
    C0 = st.slider("Initial Concentration of C (C₀)", 0.0, 10.0, value=1.0)

st.markdown("---") # Add a horizontal rule for separation

if st.button("🚀 Predict and Plot Reaction"):
    # Data Preparation for Prediction
    # Simulate the reaction using ode1 to get concentrations over time for prediction features
    time_pred, A_pred, B_pred, C_pred, k_pred, k_1_pred, is_reversible_simulated, order_simulated = ode1(A0, B0, C0, temp, Ea, A_factor)

    # Create a dictionary with all the necessary inputs for the model
    inputs = {
        'temp': temp,
        'pH': pH,
        'Ea': Ea,
        'A_factor': A_factor,
        'pressure': pressure,
        'log_pressure': np.log(pressure),
        'weight': 150,  # Using a placeholder value as it's not a user input
        'structure': structure,
        'catalyst': catalyst,
        'is_reversible': int(is_reversible),
        'k': k_pred, # Use simulated k
        'k_1': k_1_pred, # Use simulated k_1
        'A0': A_pred[0], 'A1': A_pred[1], 'A2': A_pred[2], 'A3': A_pred[3], 'A4': A_pred[4],
        'A5': A_pred[5], 'A6': A_pred[6], 'A7': A_pred[7], 'A8': A_pred[8], 'A9': A_pred[9], 'A10': A_pred[10],
        'B0': B_pred[0], 'B1': B_pred[1], 'B2': B_pred[2], 'B3': B_pred[3], 'B4': B_pred[4],
        'B5': B_pred[5], 'B6': B_pred[6], 'B7': B_pred[7], 'B8': B_pred[8], 'B9': B_pred[9], 'B10': B_pred[10],
        'C0': C_pred[0], 'C1': C_pred[1], 'C2': C_pred[2], 'C3': C_pred[3], 'C4': C_pred[4],
        'C5': C_pred[5], 'C6': C_pred[6], 'C7': C_pred[7], 'C8': C_pred[8], 'C9': C_pred[9], 'C10': C_pred[10]
    }

    # --- 2. Prediction ---
    with st.spinner('Predicting reaction order...'):
        predicted_order = predict_order(inputs)
    st.success(f"✅ Predicted Order: **{predicted_order}**")

    # --- 3. Simulation with ode2 and Predicted Order ---
    with st.spinner('Simulating reaction...'):
        time_sim, A_sim, B_sim, C_sim, k_sim, k_1_sim = ode2(A0, B0, C0, temp, Ea, A_factor, int(is_reversible), predicted_order)

    # --- 4. Plotting ---
    st.header("📊 Concentration vs. Time Plot")
    fig, ax = plt.subplots()
    ax.plot(time_sim, A_sim, label='A', marker='o') # Add markers to plot points
    ax.plot(time_sim, B_sim, label='B', marker='x')
    ax.plot(time_sim, C_sim, label='C', marker='s')
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title(f'Concentration vs. Time (Predicted Order: {predicted_order})')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

st.markdown("---")
st.markdown("App created with ❤️ by Mujtaba , Muzammil , Taha and Ali Zain.")