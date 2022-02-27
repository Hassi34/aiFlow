import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib 
from matplotlib.colors import ListedColormap
import  os 

def prepare_data(df):
  """It is used to seperate dependent(X) and independent(y) variables

  Args:
      df (pd.DataFrame): It's the pandas data frame

  Returns:
      tuple : It returns the tuples of dependent(X) and independent(y)
  """
  X = df.drop("y", axis =1)
  y = df['y']
  return X ,y 

def save_model(model, filename):
  """This method is to save the trained model with a given filename

  Args:
      model (python object): trained machine learning model
      filename (str): path to save the trained model
  """
  os.makedirs("models", exist_ok = True) #Only create if model_dir doesn't exist
  file_path = os.path.join("models", filename)
  joblib.dump(model, file_path)

def save_plot(df, file_name, model):
  """This method is used to generate and save the plot of input dataset with their seperation body generated with the model

  Args:
      df (pd.DataFrame): Pandas Dataframe having the data points to plot
      file_name (str): Name of the plot to save the file with
      model (python object): Trained model to make the decision
  """
  def _create_base_plot(df):
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()



  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)