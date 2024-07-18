from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('static/csv/final_harga_property_jateng.csv')

@app.route('/')
def index():
    # Visualisasi 1: Regresi Linear Luas Tanah vs Harga
    X1 = data['luas_tanah (m²)'].values.reshape(-1, 1)
    y1 = data['harga'].str.replace('Rp ', '').str.replace('.', '').astype(float).values.reshape(-1, 1)

    model1 = LinearRegression()
    model1.fit(X1, y1)
    y1_pred = model1.predict(X1)

    plt.figure(figsize=(6, 4))
    plt.scatter(X1, y1, color='blue')
    plt.plot(X1, y1_pred, color='red')
    plt.xlabel('Luas Tanah (m²)')
    plt.ylabel('Harga')
    plt.title('Regresi Linear Luas Tanah vs Harga')
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    plot_data1 = base64.b64encode(buffer1.getvalue()).decode()
    plt.close()

    # Visualisasi 2: Jumlah Properti berdasarkan Lokasi
    plt.figure(figsize=(6, 4))
    data['alamat'].value_counts().head(10).plot(kind='bar', color='green')
    plt.title('Jumlah Properti berdasarkan Lokasi')
    plt.xlabel('Lokasi')
    plt.ylabel('Jumlah Properti')
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    plot_data2 = base64.b64encode(buffer2.getvalue()).decode()
    plt.close()

    # Visualisasi 3: Heatmap Korelasi Dataset
    numeric_df = data.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='Pastel1')
    plt.title('Heatmap Korelasi Dataset Harga Properti')
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    plot_data3 = base64.b64encode(buffer3.getvalue()).decode()
    plt.close()

    # Render template index.html with plot images
    return render_template('index.html', plot_image1=plot_data1, plot_image2=plot_data2, plot_image3=plot_data3)

if __name__ == '__main__':
    app.run(debug=True)
