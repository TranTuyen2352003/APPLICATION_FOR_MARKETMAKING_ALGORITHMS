import itertools
import pandas as pd
import csv
import time
import SVM

def run_file():
    # if you want to try prediction only on saved model, change training to False
    training = True

    # List of basic order book features (prices and volumes for first 10 levels)
    v1 = []
    for level in range(1, 6):
        v1.extend([
            f"ask_price_{level}", f"ask_size_{level}",  # ask1_price  -->  ask_price_1  
            f"bid_price_{level}", f"bid_size_{level}"    # ask1_vol  -->  ask_size_1
        ])

    v2 = []
    for level in range(1, 6):
        v2.append(f"spread_{level}")
        v2.append(f"midprice_{level}")

    v3 = []
    for level in range(2, 6):
        v3.append(f"ask_diff_{level}")
        v3.append(f"bid_diff_{level}")

    v4 = ["avg_ask_price", "avg_bid_price", "avg_ask_vol", "avg_bid_vol"]

    v5 = ["acc_price_diff", "acc_vol_diff"]

    v6 = [
        "ask1_price_ddx",
        "bid1_price_ddx",
        "ask1_vol_ddx",
        "ask2_price_ddx",
        "bid2_price_ddx",
        "ask2_vol_ddx",
        "ask3_price_ddx",
        "bid3_price_ddx",
        "ask3_vol_ddx",
        "ask4_price_ddx",
        "bid4_price_ddx",
        "ask4_vol_ddx",
        "ask5_price_ddx",
        "bid5_price_ddx",
        "ask5_vol_ddx",
    ]

    with open("Trials.csv", "a") as csvFile:
        row = [
            "V1", "V2", "V3", "V4", "V5", "V6",
            "Accuracy", "Precision", "Recall", "Time Taken"
            #, "Filename"
        ]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # ------------ #
    #Singular Cases

    #Case V1
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3, v4, v5, v6)))
    a, p, r = SVM.main(df, "v1")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "N", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3, v4, v5, v6)))
    a, p, r = SVM.main(df, "v2")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "N", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V3
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v4, v5, v6)))
    a, p, r = SVM.main(df, "v3")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "Y", "N", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v3, v5, v6)))
    a, p, r = SVM.main(df, "v4")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "N", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4, v6)))
    a, p, r = SVM.main(df, "v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4, v5)))
    a, p, r = SVM.main(df, "v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # #Case V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "N", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    # #Case V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "N", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)


    # ------------ #
    #Double Cases V1

    #Case V1V2
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3, v4, v5, v6)))
    a, p, r = SVM.main(df, "v1v2")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "N", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V3
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v4, v5, v6)))
    a, p, r = SVM.main(df, "v1v3")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "N", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3, v5, v6)))
    a, p, r = SVM.main(df, "v1v4")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3, v4, v6)))
    a, p, r = SVM.main(df, "v1v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3, v4, v5)))
    a, p, r = SVM.main(df, "v1v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v3, v4, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v1v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "N", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v3, v4, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v1v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "N", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        

    # ------------ #
    # Double Cases V2

    # Case V2V3
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v4, v5, v6)))
    a, p, r = SVM.main(df, "v2v3")
    with open("Trials.csv", "a") as csvFile:
        row = ["N", "Y", "Y", "N", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3, v5, v6)))
    a, p, r = SVM.main(df, "v2v4")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3, v4, v6)))
    a, p, r = SVM.main(df, "v2v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3, v4, v5)))
    a, p, r = SVM.main(df, "v2v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V2V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v3, v4, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v2v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "N", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V2V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v3, v4, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v2v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "N", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        

    # ------------ #
    #Double Cases V3

    # Case V3V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v5, v6)))
    a, p, r = SVM.main(df, "v3v4")
    with open("Trials.csv", "a") as csvFile:
        row = ["N", "N", "Y", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V3V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v4, v6)))
    a, p, r = SVM.main(df, "v3v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V3V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v4, v5)))
    a, p, r = SVM.main(df, "v3v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # ------------ #
    #Double Cases V4

    #Case V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v3, v6)))
    a, p, r = SVM.main(df, "v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "N", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # Case V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v3, v5)))
    a, p, r = SVM.main(df, "v4v6")
    with open("Trials.csv", "a") as csvFile:
        row = ["N", "N", "N", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # ------------ #
    #Double Cases V5

    #Case V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4)))
    a, p, r = SVM.main(df, "v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "N", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # ------------ #
    #Triple Cases V1
    #Double Cases V2

    #Case V1V2V3
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v4, v5, v6)))
    a, p, r = SVM.main(df, "v1v2v3")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "N", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V2V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3, v5, v6)))
    a, p, r = SVM.main(df, "v1v2v4")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V2V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3, v4, v6)))
    a, p, r = SVM.main(df, "v1v2v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V2V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3, v4, v5)))
    a, p, r = SVM.main(df, "v1v2v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V2V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v3, v4, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v1v2v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "N", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V2V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v3, v4, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v1v2v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "N", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V3

    #Case V1V3V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v5, v6)))
    a, p, r = SVM.main(df, "v1v3v4")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V3V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v4, v6)))
    a, p, r = SVM.main(df, "v1v3v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V3V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v4, v5)))
    a, p, r = SVM.main(df, "v1v3v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V3V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v4, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v1v3v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "Y", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V3V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v4, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v1v3v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "Y", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V4

    #Case V1V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3, v6)))
    a, p, r = SVM.main(df, "v1v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3, v5)))
    a, p, r = SVM.main(df, "v1v4v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V4V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v3, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v1v4v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "N", "Y", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V4V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v3, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v1v4v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "N", "Y", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V5

    #Case V1V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3, v4)))
    a, p, r = SVM.main(df, "v1v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V5V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v3, v4, v6, v8)))
    # a, p, r = SVM.main(df, "v1v5v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "N", "N", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V5V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v3, v4, v6, v7)))
    # a, p, r = SVM.main(df, "v1v5v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "N", "N", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)


    # ------------ #
    #Triple Cases V2
    #Double Cases V3

    #Case V2V3V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v5, v6)))
    a, p, r = SVM.main(df, "v2v3v4")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "Y", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2V3V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v4, v6)))
    a, p, r = SVM.main(df, "v2v3v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2V3V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v4, v5)))
    a, p, r = SVM.main(df, "v2v3v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V2V3V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v4, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v2v3v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "Y", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V2V3V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v4, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v2v3v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "Y", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V4

    #Case V2V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3, v6)))
    a, p, r = SVM.main(df, "v2v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3, v5)))
    a, p, r = SVM.main(df, "v2v4v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V2V4V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v3, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v2v4v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "N", "Y", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V2V4V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v3, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v2v4v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "N", "Y", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V5

    #Case V2V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3, v4)))
    a, p, r = SVM.main(df, "v2v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V2V5V6
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v3, v4, v6, v8)))
    # a, p, r = SVM.main(df, "v2v5v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "N", "N", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V2V5V6
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v3, v4, v6, v7)))
    # a, p, r = SVM.main(df, "v2v5v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "Y", "N", "N", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)


    # ------------ #
    #Triple Cases V3

    #Double Cases V4

    #Case V3V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v6)))
    a, p, r = SVM.main(df, "v3v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "Y", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # Case V3V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v5)))
    a, p, r = SVM.main(df, "v3v4v6")
    with open("Trials.csv", "a") as csvFile:
        row = ["N", "N", "Y", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # # Case V3V4V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v3v4v7")
    # with open("Trials.csv", "a") as csvFile:
    #     row = ["N", "N", "Y", "Y", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    #     writer = csv.writer(csvFile)
    #     writer.writerow(row)
        
    # # Case V3V4V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v3v4v8")
    # with open("Trials.csv", "a") as csvFile:
    #     row = ["N", "N", "Y", "Y", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    #     writer = csv.writer(csvFile)
    #     writer.writerow(row)

    #Double Cases V5

    #Case V3V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v4)))
    a, p, r = SVM.main(df, "v3v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "Y", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V3V5V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v4, v6, v8)))
    # a, p, r = SVM.main(df, "v3v5v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "Y", "N", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V3V5V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v4, v6, v7)))
    # a, p, r = SVM.main(df, "v3v5v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "Y", "N", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)


    # ------------ #
    #Triple Cases V4
    #Double Cases V5

    #Case V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2, v3)))
    a, p, r = SVM.main(df, "v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "N", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V4V5V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v3, v6, v8)))
    # a, p, r = SVM.main(df, "v4v5v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "N", "Y", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V4V5V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v3, v6, v7)))
    # a, p, r = SVM.main(df, "v4v5v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "N", "Y", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        

    # # ------------ #
    # #Triple Cases V5
    # #Double Cases V6

    # #Case V5V6V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4, v8)))
    # a, p, r = SVM.main(df, "v5v6v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "N", "N", "Y", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V5V6V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4, v7)))
    # a, p, r = SVM.main(df, "v5v6v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "N", "N", "Y", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)


    # # ------------ #
    # #Triple Cases V6

    # #Case V6V7V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v1, v2, v3, v4, v5)))
    # a, p, r = SVM.main(df, "v6v7v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["N", "N", "N", "N", "N", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)


    # ------------ #
    #Quadruple Cases V1
    #Triple Cases V2
    #Double Cases V3

    #Case V1V2V3V4
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v5, v6)))
    a, p, r, = SVM.main(df, "v1v2v3v4")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V2V3V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v4, v6)))
    a, p, r, = SVM.main(df, "v1v2v3v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V2V3V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v4, v5)))
    a, p, r = SVM.main(df, "v1v2v3v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V2V3V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v4, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v1v2v3v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "Y", "N", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V2V3V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v4, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v1v2v3v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "Y", "N", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V4

    #Case V1V2V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3, v6)))
    a, p, r = SVM.main(df, "v1v2v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V2V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3, v5)))
    a, p, r = SVM.main(df, "v1v2v4v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V2V4V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v3, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v1v2v4v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "N", "Y", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V2V4V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v3, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v1v2v4v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "N", "Y", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V5

    #Case V1V2V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3, v4)))
    a, p, r = SVM.main(df, "v1v2v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V2V5V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v3, v4, v6, v8)))
    # a, p, r = SVM.main(df, "v1v2v5v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "N", "N", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V2V5V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v3, v4, v6, v7)))
    # a, p, r = SVM.main(df, "v1v2v5v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "Y", "N", "N", "Y", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Triple Cases V3
    #Double Cases V4

    #Case V1V3V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v6)))
    a, p, r = SVM.main(df, "v1v3v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V3V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v5)))
    a, p, r = SVM.main(df, "v1v3v4v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V3V4V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v5, v6, v8)))
    # a, p, r = SVM.main(df, "v1v3v4v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "Y", "Y", "N", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V3V4V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v5, v6, v7)))
    # a, p, r = SVM.main(df, "v1v3v4v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "Y", "Y", "N", "N", "N", "Y", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Double Cases V5

    #Case V1V3V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v4)))
    a, p, r = SVM.main(df, "v1v3v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V3V5V7
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v4, v6, v8)))
    # a, p, r = SVM.main(df, "v1v3v5v7")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "Y", "N", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)
        
    # #Case V1V3V5V8
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v4, v6, v7)))
    # a, p, r = SVM.main(df, "v1v3v5v8")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "Y", "N", "Y", "N", "Y", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)

    #Triple Cases V4
    #Double Cases V5

    #Case V1V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3)))
    a, p, r = SVM.main(df, "v1v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    #Case V1V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2, v3)))
    a, p, r = SVM.main(df, "v1v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "N", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        
    # #Case V1V4V5V6
    # start_time = time.time()
    # df = pd.read_csv(r'processed_orderbook_dfm.csv')
    # df = df.drop(columns=list(itertools.chain(v2, v3, v7, v8)))
    # a, p, r = SVM.main(df, "v1v4v5v6")
    # with open('Trials.csv', 'a') as csvFile:
    # 	row = ["Y", "N", "N", "Y", "Y", "Y", "N", "N", a, p, r, str(time.time() - start_time)]
    # 	writer = csv.writer(csvFile)
    # 	writer.writerow(row)


    # ------------ #
    #Quadruple Cases V2
    #Triple Cases V3
    #Double Cases V4

    #Case V2V3V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v6)))
    a, p, r = SVM.main(df, "v2v3v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "Y", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V2V3V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v5)))
    a, p, r = SVM.main(df, "v2v3v4v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "Y", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Double Cases V5

    #Case V2V3V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v4)))
    a, p, r = SVM.main(df, "v2v3v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "Y", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Triple Cases V4
    #Double Cases V5

    #Case V2V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v3)))
    a, p, r = SVM.main(df, "v2v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "N", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)


    # ------------ #
    #Quadruple Cases V3
    #Triple Cases V4
    #Double Cases V5

    #Case V3V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1, v2)))
    a, p, r = SVM.main(df, "v3v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "N", "Y", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Quintuple Cases V1
    #Quadruple Cases V2
    #Triple Cases V3
    #Double Cases V4

    #Case V1V2V3V4V5
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v6)))
    a, p, r = SVM.main(df, "v1v2v3v4v5")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "Y", "Y", "N", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Case V1V2V3V4V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v5)))
    a, p, r = SVM.main(df, "v1v2v3v4v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "Y", "N", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Double Cases V5

    #Case V1V2V3V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v4)))
    a, p, r = SVM.main(df, "v1v2v3v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "N", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Triple Cases V4
    #Double Cases V5

    #Case V1V2V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v3)))
    a, p, r = SVM.main(df, "v1v2v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "N", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Quadruple Cases V3
    #Triple Cases V4
    #Double Cases V5

    #Case V1V3V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v2)))
    a, p, r = SVM.main(df, "v1v3v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "N", "Y", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Quintuple Cases V2
    #Quadruple Cases V3
    #Triple Cases V4
    #Double Cases V5

    #Case V2V3V4V5V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    df = df.drop(columns=list(itertools.chain(v1)))
    a, p, r, = SVM.main(df, "v2v3v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["N", "Y", "Y", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)

    #Final Case V1-V6
    start_time = time.time()
    df = pd.read_csv(r'processed_orderbook_dfm.csv')
    a, p, r = SVM.main(df, "v1v2v3v4v5v6")
    with open('Trials.csv', 'a') as csvFile:
        row = ["Y", "Y", "Y", "Y", "Y", "Y", a, p, r, str(time.time() - start_time)]
        writer = csv.writer(csvFile)
        writer.writerow(row)
	