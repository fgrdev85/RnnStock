def preparar_datos(secuencia, pasos):
  X, y = [], []
  for i in range(len(secuencia)-(pasos+2)):
    #print(secuencia[i:i+pasos])
    #print(secuencia[i+2:i+pasos+2])
    #print(' ')
    X.append(secuencia[i:i+pasos])
    y.append(secuencia[i+2:i+pasos+2])  # Predicción de múltiples valores futuros
  return np.array(X), np.array(y)

contador = 1
#df_stocks = pd.read_csv('/content/drive/MyDrive/datosRNNsimple/spstocks.csv')
df_stocks = pd.read_csv('/content/drive/MyDrive/datosRNNsimple/StocksDemo.csv')
for i in df_stocks.Clave:
        print(i, contador)
        print(strftime("%a, %d %b %Y %H:%M:%S", gmtime()))
        print('Entrenando modelo...')
        df = pd.read_csv('/content/drive/MyDrive/datosRNNsimple/stocks/'+i+'_out.csv')
        #print(df)
        secuencia_2 = df['Close'].to_numpy()
        #print(secuencia_2)
        pasos = 16
        X, y = preparar_datos(secuencia_2, pasos)
        #print(' X ',X)
        #print(' y ', y)

        # Construir modelo RNN
        modelo = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(50, activation='relu', input_shape=(pasos, 1), return_sequences=True),
          tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
          tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
          tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        ])



        modelo.compile(optimizer='adam', loss='mse')

        # Ajustar modelo
        X = X.reshape((X.shape[0], X.shape[1], 1))
        modelo.fit(X, y, epochs=100, verbose=0)
        modelo.save('/content/drive/MyDrive/datosRNNsimple/Preentrenados/pretrain_new_input_len_16/'+i+'_out.keras')
        print('Ok')
        contador += 1
        print(strftime("%a, %d %b %Y %H:%M:%S", gmtime()))
        print(' ')
