from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import logging, os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename = os.path.join(log_dir,"running_logs.log"), level= logging.INFO, format=logging_str, filemode="a")
def main(data, eta, epochs, filename, plot_filename):
    df = pd.DataFrame(data)
    logging.info(f"This is the actual dataframe{df}")
    X , y = prepare_data(df)
    model = Perceptron(eta=eta, epochs= epochs)
    model.fit(X, y)
    _ = model.total_loss()

    save_model(model, filename)
    save_plot(df, plot_filename , model)
if __name__ == '__main__':
    OR = {
        'x1' : [0,0,1,1],
        'x2' : [0,1,0, 1],
        'y' : [0, 1,1,1]
    }
    ETA = 0.3
    EPOCHS = 100
    try:
        logging.info(">>>>>>>>> Starting Training >>>>>>>>>>>")
        main(OR, ETA, EPOCHS, "or.model", "or.png")
        logging.info(">>>>>>>>> Training Ended with Success >>>>>>>>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e