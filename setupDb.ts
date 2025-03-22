import sqlite3 from "sqlite3";
import { open } from "sqlite";

// Veritabanı bağlantısı
export async function getDb() {
    return open({
        filename: "./database.sqlite",
        driver: sqlite3.Database,
    });
}

// Veritabanını başlat
async function initializeDb() {
    const db = await getDb();

    // Tabloları oluştur
    await db.exec(`
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories (id)
        );

        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            FOREIGN KEY (topic_id) REFERENCES topics (id)
        );
    `);

    console.log("Veritabanı ve tablolar hazır.");
}

// Veriyi ekle
async function insertData() {
    const db = await getDb();

    const category = {
        name: "Python",
        topics: [
                {
                        name: "Python",
                        questions: [
                            { question: "Python nedir?", answer: "Python, yüksek seviyeli, yorumlamalı ve genel amaçlı bir programlama dilidir." },
                            { question: "Python'da bir değişken nasıl tanımlanır?", answer: "Değişken adı ve değer ataması yapılarak tanımlanır, örneğin: x = 5" },
                            { question: "Python'da bir liste nasıl oluşturulur?", answer: "Köşeli parantezler kullanılarak oluşturulur, örneğin: my_list = [1, 2, 3]" },
                            { question: "Python'da bir fonksiyon nasıl tanımlanır?", answer: "def anahtar kelimesi kullanılarak tanımlanır, örneğin:\n" +
                                    "def my_function():\n" +
                                    "    print(\"Hello, World!\")" },
                            { question: "Python'da bir döngü nasıl oluşturulur?", answer: "for veya while döngüleri kullanılarak oluşturulur, örneğin:\n" +
                                    "for i in range(5):\n" +
                                    "    print(i)" },
                            { question: "Python'da sınıf (class) nasıl tanımlanır ve kullanılır?", answer: "class anahtar kelimesi kullanılarak tanımlanır ve örneklenir, örneğin:\n" +
                                    "class MyClass:\n" +
                                    "    def __init__(self, name):\n" +
                                    "        self.name = name\n" +
                                    "\n" +
                                    "    def greet(self):\n" +
                                    "        print(f\"Hello, {self.name}!\")\n" +
                                    "\n" +
                                    "obj = MyClass(\"Alice\")\n" +
                                    "obj.greet()" },
                            { question: "Python'da liste anlama (list comprehension) nedir ve nasıl kullanılır?", answer: "Liste oluşturmanın kısa bir yoludur, örneğin:\n" +
                                    "squares = [x**2 for x in range(10)]" },
                            { question: "Python'da hata yönetimi (exception handling) nasıl yapılır?", answer: "try, except, else ve finally blokları kullanılarak yapılır, örneğin:\n" +
                                    "try:\n" +
                                    "    result = 10 / 0\n" +
                                    "except ZeroDivisionError:\n" +
                                    "    print(\"Sıfıra bölme hatası!\")\n" +
                                    "else:\n" +
                                    "    print(\"İşlem başarılı.\")\n" +
                                    "finally:\n" +
                                    "    print(\"İşlem tamamlandı.\")" },
                            { question: "Python'da dekoratör (decorator) nedir ve nasıl kullanılır?", answer: "Fonksiyonların davranışını değiştirmek için kullanılan fonksiyonlardır, örneğin:\n" +
                                    "def my_decorator(func):\n" +
                                    "    def wrapper():\n" +
                                    "        print(\"Fonksiyondan önce\")\n" +
                                    "        func()\n" +
                                    "        print(\"Fonksiyondan sonra\")\n" +
                                    "    return wrapper\n" +
                                    "\n" +
                                    "@my_decorator\n" +
                                    "def say_hello():\n" +
                                    "    print(\"Hello!\")\n" +
                                    "\n" +
                                    "say_hello()" },
                            { question: "Python'da jeneratör (generator) nedir ve nasıl kullanılır?", answer: "Bellek verimliliği için kullanılan özel fonksiyonlardır, yield ifadesiyle değer döndürürler, örneğin:\n" +
                                    "def count_up_to(n):\n" +
                                    "    count = 1\n" +
                                    "    while count <= n:\n" +
                                    "        yield count\n" +
                                    "        count += 1\n" +
                                    "\n" +
                                    "for number in count_up_to(3):\n" +
                                    "    print(number)" },
        
                        ],
                    },
                    {
                        name: "NumPy",
                        questions: [
                            { question: "NumPy nedir?", answer: "NumPy, bilimsel hesaplama için kullanılan bir Python kütüphanesidir." },
                            { question: "NumPy kütüphanesi nasıl yüklenir?", answer: "pip install numpy komutu ile yüklenir.\n" },
                            { question: "NumPy'de bir dizi (array) nasıl oluşturulur?", answer: "numpy.array() fonksiyonu kullanılarak oluşturulur, örneğin: np.array([1, 2, 3])." },
                            { question: "NumPy'de rastgele sayılar içeren bir dizi nasıl oluşturulur?", answer: "numpy.random modülü kullanılır, örneğin: np.random.rand(3, 3) 3x3'lük bir rastgele sayı matrisi oluşturur.\n" },
                            { question: "NumPy'de bir dizinin tipi nasıl öğrenilir?", answer: "array.dtype özelliği ile öğrenilir." },
                            { question: "NumPy'de bir dizideki eksik (NaN) değerler nasıl bulunur ve doldurulur?", answer: "numpy.isnan() fonksiyonu ile bulunur, doldurmak için numpy.nan_to_num() kullanılabilir:\n" +
                                    "array = np.array([1, np.nan, 3])\n" +
                                    "np.nan_to_num(array)  # [1. 0. 3.]" },
                            { question: "NumPy hangi alanlarda kullanılır?", answer: "NumPy, veri analizi, bilimsel hesaplama, makine öğrenimi ve görüntü işleme gibi alanlarda kullanılır." },
                            { question: "NumPy'de bir diziyi belirli bir eksene göre sıralama nasıl yapılır?", answer: "numpy.sort() fonksiyonu ve axis parametresi kullanılır:\n" +
                                    "array = np.array([[3, 2, 1], [6, 5, 4]])\n" +
                                    "sorted_array = np.sort(array, axis=1)  # Sırala" },
                            { question: "NumPy dizilerinin boyutu nasıl öğrenilir?", answer: "array.shape özelliği ile öğrenilir." },
                            { question: "NumPy, veri görselleştirme kütüphaneleriyle nasıl entegre edilir?", answer: "NumPy dizileri, Matplotlib ve Seaborn gibi kütüphanelerle grafik oluşturmak için kullanılabilir." },
                        ],
                    },
                    {
                        name: "Pandas",
                        questions: [
                            { question: "Pandas nedir?", answer: "Pandas, Python'da veri analizi ve manipülasyonu için kullanılan bir kütüphanedir." },
                            { question: "Pandas kütüphanesi nasıl yüklenir?", answer: "pip install pandas komutu ile yüklenir.\n" },
                            { question: "Pandas'ta bir DataFrame nedir?", answer: "DataFrame, tablo benzeri iki boyutlu bir veri yapısıdır." },
                            { question: "Pandas'ta bir CSV dosyası nasıl okunur?", answer: "pandas.read_csv('dosya_adi.csv') ile okunur.\n" },
                            { question: "Pandas'ta eksik (NaN) değerler nasıl tespit edilir ve doldurulur?", answer: "isna() veya isnull() ile tespit edilir, fillna() ile doldurulur, örneğin\n" +
                            "df['sutun'] = df['sutun'].fillna(0)"},
                            { question: "Bir DataFrame’deki belirli bir koşula uyan satırları nasıl seçersiniz?", answer: "loc[] veya query() kullanılır:\n" +
                                    "filtered = df.loc[df['sutun'] > 50]" },
                            { question: "Pandas'ta gruplama işlemleri nasıl yapılır ve her grubun ortalaması nasıl alınır?", answer: "groupby() ve mean() kullanılarak yapılır:\n" +
                                    "grouped = df.groupby('kategori_sutun')['deger_sutun'].mean()" },
                            { question: "Pandas ile birden fazla DataFrame nasıl birleştirilir?", answer: "merge() veya concat() kullanılır:\n" +
                                    "result = pd.merge(df1, df2, on='ortak_sutun', how='inner')" },
                            { question: "Pandas'ta pivot tablolar nasıl oluşturulur?", answer: "pivot_table() fonksiyonu ile oluşturulur:\n" +
                                    "pivot = df.pivot_table(values='deger', index='kategori', columns='yil', aggfunc='sum')" },
                            { question: "Pandas'ta bir sütundaki benzersiz değerler nasıl bulunur?", answer: "dataframe['sutun_adi'].unique() fonksiyonu ile bulunur." },
                        ],
                    },
                    {
                        name: "SciPy",
                        questions: [
                            { question: "SciPy nedir?", answer: "SciPy, bilimsel ve teknik hesaplamalar için kullanılan bir Python kütüphanesidir." },
                            { question: "SciPy kütüphanesi nasıl yüklenir?", answer: "pip install scipy komutu ile yüklenir." },
                            { question: "SciPy’nin temel modülleri nelerdir?", answer: "SciPy'nin temel modülleri: scipy.integrate, scipy.optimize, scipy.stats, scipy.linalg, ve scipy.spatialdır.\n" },
                            { question: "SciPy ile bir fonksiyonun kökü nasıl bulunur?", answer: "scipy.optimize.root() fonksiyonu ile bulunur." },
                            { question: "SciPy'de bir matrisin determinantı nasıl hesaplanır?", answer: "scipy.linalg.det() fonksiyonu ile hesaplanır." },
                            { question: "SciPy ile bir fonksiyonun belirli bir aralıktaki integralini nasıl hesaplayabilirsiniz?", answer: "scipy.integrate.quad() fonksiyonu ile hesaplanır:\n" +
                                    "from scipy.integrate import quad\n" +
                                    "result, error = quad(lambda x: x**2, 0, 2)" },
                            { question: "SciPy kullanarak bir veri kümesi için histogram dağılımı nasıl hesaplanır?", answer: "scipy.stats modülündeki binned_statistic fonksiyonu ile yapılır:\n" +
                                    "from scipy.stats import binned_statistic\n" +
                                    "histogram, edges, _ = binned_statistic(data, data, statistic='count', bins=10)" },
                            { question: "SciPy ile doğrusal cebir problemleri nasıl çözülür? Örneğin, bir Ax = B denklemi.", answer: "scipy.linalg.solve() fonksiyonu kullanılır:\n" +
                                    "from scipy.linalg import solve\n" +
                                    "A = np.array([[3, 2], [1, 4]])\n" +
                                    "B = np.array([5, 6])\n" +
                                    "x = solve(A, B)" },
                            { question: "SciPy ile optimize edilmiş bir fonksiyonun minimum değeri nasıl bulunur?", answer: "scipy.optimize.minimize() fonksiyonu ile yapılır:\n" +
                                    "from scipy.optimize import minimize\n" +
                                    "result = minimize(lambda x: x**2 + 2*x + 1, x0=0)" },
                            { question: "SciPy'nin Fourier dönüşümü nasıl uygulanır ve bu dönüşüm ile bir sinyalin frekans spektrumu nasıl elde edilir?", answer: "scipy.fft modülü ile yapılır:\n" +
                                    "from scipy.fft import fft\n" +
                                    "signal = np.array([1, 2, 3, 4])\n" +
                                    "spectrum = fft(signal)" },
                        ],
                    },
                    {
                        name: "Matplotlib & Seaborn",
                        questions: [
                            { question: "Matplotlib nedir ve ne için kullanılır?", answer: "Matplotlib, veri görselleştirme ve grafik oluşturma için kullanılan bir Python kütüphanesidir." },
                            { question: "Seaborn nedir ve neden Matplotlib'e ek olarak kullanılır?", answer: "Seaborn, Matplotlib üzerine kurulu, daha estetik ve kolay veri görselleştirme sağlar.\n" },
                            { question: "Matplotlib ile bir çizgi grafiği nasıl oluşturulur?", answer: "plt.plot() fonksiyonu kullanılır:\n" +
                                    "import matplotlib.pyplot as plt\n" +
                                    "plt.plot([1, 2, 3], [4, 5, 6])\n" +
                                    "plt.show()" },
                            { question: "Seaborn ile bir dağılım grafiği (scatterplot) nasıl çizilir?", answer: "sns.scatterplot() fonksiyonu kullanılır:\n" +
                                    "import seaborn as sns\n" +
                                    "sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])" },
                            { question: "Matplotlib ve Seaborn grafikleri aynı figürde birleştirilebilir mi?", answer: "Evet, aynı figürde Matplotlib ve Seaborn kullanılabilir." },
                            { question: "Matplotlib ile eksenleri özelleştirme nasıl yapılır? Örneğin, başlık ekleme, etiket koyma, ve aralık ayarlama.", answer: "plt.title(), plt.xlabel(), plt.ylabel() ve plt.axis() kullanılır:\n" +
                                    "plt.plot([1, 2, 3], [4, 5, 6])\n" +
                                    "plt.title(\"Grafik Başlığı\")\n" +
                                    "plt.xlabel(\"X Ekseni\")\n" +
                                    "plt.ylabel(\"Y Ekseni\")\n" +
                                    "plt.axis([0, 4, 3, 7])  # Aralık: [xmin, xmax, ymin, ymax]\n" +
                                    "plt.show()" },
                            { question: "Seaborn'da bir veri çerçevesini kullanarak gruplara göre bir bar grafiği nasıl oluşturulur?", answer: "sns.barplot() kullanılır:\n" +
                                    "import seaborn as sns\n" +
                                    "import pandas as pd\n" +
                                    "data = pd.DataFrame({'Grup': ['A', 'B', 'C'], 'Değer': [10, 20, 15]})\n" +
                                    "sns.barplot(x='Grup', y='Değer', data=data)" },
                            { question: "Matplotlib'de çoklu alt grafik (subplot) nasıl oluşturulur?", answer: "plt.subplot() kullanılarak birden fazla grafik aynı figürde gösterilebilir:\n" +
                                    "plt.subplot(2, 1, 1)  # 2 satır, 1 sütun, 1. grafik\n" +
                                    "plt.plot([1, 2, 3], [4, 5, 6])\n" +
                                    "plt.subplot(2, 1, 2)  # 2. grafik\n" +
                                    "plt.plot([1, 2, 3], [6, 5, 4])\n" +
                                    "plt.show()" },
                            { question: "Seaborn ile bir korelasyon matrisini görselleştirmek için hangi yöntem kullanılır?", answer: "sns.heatmap() fonksiyonu ile görselleştirilir:\n" +
                                    "import numpy as np\n" +
                                    "import seaborn as sns\n" +
                                    "import pandas as pd\n" +
                                    "data = pd.DataFrame(np.random.rand(5, 5), columns=list('ABCDE'))\n" +
                                    "sns.heatmap(data.corr(), annot=True)" },
                            { question: "Matplotlib ve Seaborn’da renk paletlerini özelleştirme nasıl yapılır? Örneğin, bir grafik için özel bir renk paleti seçmek.", answer: "Seaborn'da sns.set_palette(), Matplotlib'de color parametresi kullanılır:\n" +
                                    "sns.set_palette(\"husl\")\n" +
                                    "sns.barplot(x=\"Grup\", y=\"Değer\", data=data, ci=None)" },
                        ],
                    },
                    {
                        name: "Scikit-Learn",
                        questions: [
                            { question: "Scikit-learn nedir?", answer: "Scikit-learn, Python'da makine öğrenimi algoritmalarını uygulamak için kullanılan bir kütüphanedir." },
                            { question: "Scikit-learn ile hangi tür makine öğrenimi algoritmalarını uygulayabilirsiniz?", answer: "Sınıflandırma, regresyon, kümeleme, boyut indirgeme, model seçimi ve ön işleme algoritmalarını içerir." },
                            { question: "Scikit-learn kütüphanesini yüklemek için hangi komut kullanılır?", answer: "pip install scikit-learn komutu ile yüklenir." },
                            { question: "Scikit-learn'de veriyi eğitmek için hangi fonksiyon kullanılır?", answer: ".fit() fonksiyonu, modelin veriye uyarlanması için kullanılır." },
                            { question: "Scikit-learn'de bir modelin tahmin yapması için hangi fonksiyon kullanılır?", answer: ".predict() fonksiyonu ile model tahmin yapar." },
                            { question: "Scikit-learn ile bir sınıflandırma modelini nasıl oluşturabilirsiniz? Örneğin, K-Nearest Neighbors (KNN).", answer: "KNeighborsClassifier ile model oluşturulur ve veriye .fit() ile eğitilir:\n" +
                                    "from sklearn.neighbors import KNeighborsClassifier\n" +
                                    "model = KNeighborsClassifier(n_neighbors=3)\n" +
                                    "model.fit(X_train, y_train)" },
                            { question: "Scikit-learn’de bir modelin başarımını değerlendirmek için hangi metrikler kullanılır?", answer: "accuracy_score, confusion_matrix, precision_score, recall_score, f1_score gibi metrikler kullanılır. Örneğin:\n" +
                                    "from sklearn.metrics import accuracy_score\n" +
                                    "y_pred = model.predict(X_test)\n" +
                                    "accuracy = accuracy_score(y_test, y_pred)" },
                            { question: "Scikit-learn ile veriyi normalleştirmek veya ölçeklendirmek için hangi işlemler yapılır?", answer: "StandardScaler veya MinMaxScaler kullanılır. Örneğin:\n" +
                                    "from sklearn.preprocessing import StandardScaler\n" +
                                    "scaler = StandardScaler()\n" +
                                    "X_scaled = scaler.fit_transform(X)" },
                            { question: "Scikit-learn ile çapraz doğrulama (cross-validation) nasıl yapılır?", answer: "cross_val_score veya cross_validate fonksiyonları kullanılır:\n" +
                                    "from sklearn.model_selection import cross_val_score\n" +
                                    "scores = cross_val_score(model, X, y, cv=5)" },
                            { question: "Scikit-learn'de regresyon modeli (örneğin, doğrusal regresyon) nasıl oluşturulur ve tahmin yapılır?", answer: "LinearRegression kullanılarak model oluşturulur ve eğitilir:\n" +
                                    "from sklearn.linear_model import LinearRegression\n" +
                                    "model = LinearRegression()\n" +
                                    "model.fit(X_train, y_train)\n" +
                                    "y_pred = model.predict(X_test)" },
                        ],
                    },
                    {
                        name: "TensorFlow & PyTorch",
                        questions: [
                            { question: "TensorFlow nedir?", answer: "TensorFlow, makine öğrenimi ve derin öğrenme uygulamaları geliştirmek için kullanılan açık kaynaklı bir kütüphanedir." },
                            { question: "PyTorch nedir ve hangi tür uygulamalar için kullanılır?", answer: "PyTorch, dinamik hesaplama grafikleriyle derin öğrenme uygulamaları geliştirmek için kullanılan açık kaynaklı bir kütüphanedir." },
                            { question: "TensorFlow ve PyTorch arasındaki temel farklar nelerdir?", answer: "TensorFlow, statik hesaplama grafiği kullanırken, PyTorch dinamik hesaplama grafiği kullanır." },
                            { question: "TensorFlow ile bir model nasıl oluşturulur?", answer: "tf.keras API'si kullanılarak model oluşturulur:\n" +
                                    "import tensorflow as tf\n" +
                                    "model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])\n" +
                                    "model.compile(optimizer='adam', loss='mse')" },
                            { question: "PyTorch ile bir model nasıl eğitilir?", answer: "PyTorch’ta modelin .forward() fonksiyonu kullanılarak eğitim yapılır." },
                            { question: "TensorFlow’da bir sinir ağı modeli nasıl tanımlanır ve eğitim verisi ile nasıl eğitilir?", answer: "tf.keras.Sequential veya tf.keras.Model kullanılarak model tanımlanır, ardından .fit() fonksiyonu ile eğitim yapılır:\n" +
                                    "model = tf.keras.Sequential([\n" +
                                    "    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),\n" +
                                    "    tf.keras.layers.Dense(10, activation='softmax')\n" +
                                    "])\n" +
                                    "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n" +
                                    "model.fit(x_train, y_train, epochs=10)" },
                            { question: "PyTorch ile bir modelin kaybı nasıl hesaplanır ve geri yayılım nasıl yapılır?", answer: "torch.nn.CrossEntropyLoss() kayıp fonksiyonu ile kayıp hesaplanır ve .backward() fonksiyonu ile geri yayılım yapılır:\n" +
                                    "import torch\n" +
                                    "import torch.nn as nn\n" +
                                    "loss_fn = nn.CrossEntropyLoss()\n" +
                                    "output = model(inputs)\n" +
                                    "loss = loss_fn(output, labels)\n" +
                                    "loss.backward()  # Geri yayılım\n" +
                                    "optimizer.step()  # Parametre güncelleme" },
                            { question: "TensorFlow'da bir modelin doğruluğu nasıl değerlendirilir", answer: "model.evaluate() fonksiyonu ile doğruluk hesaplanır:\n" +
                                    "test_loss, test_accuracy = model.evaluate(x_test, y_test)" },
                            { question: "PyTorch’ta bir modelin eğitim döngüsünü nasıl yazarsınız?", answer: "Eğitim döngüsünde veri, kayıp hesaplama, geri yayılım ve optimizasyon adımları yapılır:\n" +
                                    "for epoch in range(num_epochs):\n" +
                                    "    for inputs, labels in train_loader:\n" +
                                    "        optimizer.zero_grad()\n" +
                                    "        outputs = model(inputs)\n" +
                                    "        loss = loss_fn(outputs, labels)\n" +
                                    "        loss.backward()\n" +
                                    "        optimizer.step()" },
                            { question: "TensorFlow ve PyTorch'ta GPU kullanımı nasıl yapılır?", answer: "TensorFlow’da tf.device() ile, PyTorch’ta torch.cuda.is_available() ile GPU kullanılabilir:\n" +
                                    "# TensorFlow\n" +
                                    "with tf.device('/GPU:0'):\n" +
                                    "    model.fit(x_train, y_train)\n" +
                                    "# PyTorch\n" +
                                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n" +
                                    "model.to(device)" },
                        ],
                    },
                    {
                        name: "Flask",
                        questions: [
                            { question: "Flask nedir?", answer: "Flask, Python ile web uygulamaları geliştirmek için kullanılan hafif, esnek bir mikro web framework'üdür." },
                            { question: "Flask ile web uygulaması geliştirmek için hangi modül import edilir?", answer: "from flask import Flask modülü ile Flask import edilir" },
                            { question: "Flask'ta bir rota (route) nasıl tanımlanır?", answer: "@app.route() dekoratörü kullanılarak bir rota tanımlanır:\n" +
                                    "@app.route('/')\n" +
                                    "def home():\n" +
                                    "    return 'Merhaba, Dünya!'" },
                            { question: "Flask'ta HTTP GET ve POST istekleri nasıl yönetilir?", answer: "Rota dekoratöründe methods parametresi kullanılarak GET ve POST istekleri yönetilir:\n" +
                                    "@app.route('/form', methods=['GET', 'POST'])\n" +
                                    "def form():\n" +
                                    "    if request.method == 'POST':\n" +
                                    "        return 'Form gönderildi!'\n" +
                                    "    return render_template('form.html')" },
                            { question: "Flask uygulaması nasıl çalıştırılır?", answer: "app.run() komutu ile Flask uygulaması çalıştırılır:\n" +
                                    "if __name__ == '__main__':\n" +
                                    "    app.run(debug=True)" },
                            { question: "Flask'ta birden fazla rota tanımlaması nasıl yapılır ve farklı HTTP metodları nasıl yönetilir?", answer: "Birden fazla rota tanımlamak için birden fazla @app.route() dekoratörü kullanılır. Ayrıca, methods parametresi ile HTTP metodları belirtilir:\n" +
                                    "@app.route('/get', methods=['GET'])\n" +
                                    "def get_route():\n" +
                                    "    return 'GET isteği alındı!'\n" +
                                    "\n" +
                                    "@app.route('/post', methods=['POST'])\n" +
                                    "def post_route():\n" +
                                    "    return 'POST isteği alındı!'" },
                            { question: "Flask'ta form verilerini nasıl alırsınız ve işlersiniz?", answer: "Flask'ta form verilerini almak için request.form kullanılır:\n" +
                                    "from flask import request\n" +
                                    "@app.route('/submit', methods=['POST'])\n" +
                                    "def submit():\n" +
                                    "    name = request.form['name']\n" +
                                    "    return f'Form verisi: {name}'" },
                            { question: "Flask'ta bir HTML dosyasını nasıl render edersiniz?", answer: "Flask'ta HTML dosyasını render_template() fonksiyonu ile render edersiniz:\n" +
                                    "from flask import render_template\n" +
                                    "@app.route('/')\n" +
                                    "def home():\n" +
                                    "    return render_template('index.html')" },
                            { question: "Flask'ta session yönetimi nasıl yapılır?", answer: "Flask'ta kullanıcı oturumları için session kullanılır. app.secret_key ile güvenlik sağlanır:\n" +
                                    "from flask import session\n" +
                                    "app.secret_key = 'secretkey'\n" +
                                    "@app.route('/set_session')\n" +
                                    "def set_session():\n" +
                                    "    session['user'] = 'John Doe'\n" +
                                    "    return 'Session set!'\n" +
                                    "\n" +
                                    "@app.route('/get_session')\n" +
                                    "def get_session():\n" +
                                    "    return f\"User: {session.get('user')}\"" },
                            { question: "Flask uygulamasında hata yönetimi nasıl yapılır?", answer: "Flask'ta hata yönetimi için @app.errorhandler() dekoratörü kullanılır:\n" +
                                    "@app.errorhandler(404)\n" +
                                    "def not_found(error):\n" +
                                    "    return 'Sayfa bulunamadı', 404\n" +
                                    "\n" +
                                    "@app.errorhandler(500)\n" +
                                    "def internal_error(error):\n" +
                                    "    return 'Sunucu hatası', 500" },
                        ],
                    },
                    {
                        name: "OpenCV",
                        questions: [
                            { question: "OpenCV nedir?", answer: "OpenCV, görsel işleme ve bilgisayarla görme (computer vision) uygulamaları geliştirmek için kullanılan açık kaynaklı bir kütüphanedir" },
                            { question: "OpenCV ile bir resmi nasıl okursunuz?", answer: "cv2.imread() fonksiyonu ile bir resmi okursunuz:\n" +
                                    "import cv2\n" +
                                    "img = cv2.imread('image.jpg')" },
                            { question: "OpenCV ile bir resmi nasıl gösterirsiniz?", answer: "cv2.imshow() fonksiyonu ile bir resmi ekranda gösterirsiniz:\n" +
                                    "cv2.imshow('Resim', img)\n" +
                                    "cv2.waitKey(0)\n" +
                                    "cv2.destroyAllWindows()" },
                            { question: "OpenCV ile bir resmi gri tonlara nasıl dönüştürürsünüz?", answer: "cv2.cvtColor() fonksiyonu ile resmi gri tonlara dönüştürürsünüz:\n" +
                                    "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)" },
                            { question: "OpenCV ile bir resmin boyutlarını nasıl öğrenirsiniz?", answer: "img.shape ile resmin boyutlarını öğrenirsiniz:\n" +
                                    "height, width, channels = img.shape" },
                            { question: "OpenCV ile bir görüntüyü nasıl kırparsınız?", answer: "Görüntü kırpma işlemi, dizin dilimleme yöntemi ile yapılır:\n" +
                                    "cropped_img = img[100:400, 150:500]  # Yatay ve dikey kırpma\n" +
                                    "cv2.imshow('Kırpılmış Resim', cropped_img)\n" +
                                    "cv2.waitKey(0)" },
                            { question: "OpenCV ile bir resmi nasıl kaydedersiniz?", answer: "cv2.imwrite() fonksiyonu ile resmi kaydedebilirsiniz:\n" +
                                    "cv2.imwrite('saved_image.jpg', img)" },
                            { question: "OpenCV ile bir resmi nasıl bulanıklaştırırsınız?", answer: "cv2.GaussianBlur() fonksiyonu ile bir resmi bulanıklaştırabilirsiniz:\n" +
                                    "blurred_img = cv2.GaussianBlur(img, (15, 15), 0)" },
                            { question: "OpenCV ile bir resme kenar algılama nasıl yapılır?", answer: "cv2.Canny() fonksiyonu ile kenar algılama yapılabilir:\n" +
                                    "edges = cv2.Canny(img, 100, 200)\n" +
                                    "cv2.imshow('Kenar Algılama', edges)\n" +
                                    "cv2.waitKey(0)" },
                            { question: "OpenCV ile bir görüntüye metin nasıl eklenir?", answer: "cv2.putText() fonksiyonu ile bir görüntüye metin eklenebilir:\n" +
                                    "font = cv2.FONT_HERSHEY_SIMPLEX\n" +
                                    "cv2.putText(img, 'Merhaba OpenCV!', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)\n" +
                                    "cv2.imshow('Metinli Resim', img)\n" +
                                    "cv2.waitKey(0)" },
                        ],
                    },
                    {
                        name: "Selenium",
                        questions: [
                            { question: "Selenium nedir?", answer: "Selenium, web tarayıcılarını otomatikleştirmek için kullanılan açık kaynaklı bir kütüphanedir." },
                            { question: "Selenium ile hangi programlama dillerinde test yazılabilir?", answer: "Selenium, Python, Java, C#, Ruby gibi dillerde kullanılabilir." },
                            { question: "Selenium ile bir web sayfasını nasıl açarsınız?", answer: "webdriver kullanarak tarayıcıyı başlatıp sayfayı açabilirsiniz:\n" +
                                    "from selenium import webdriver\n" +
                                    "driver = webdriver.Chrome()\n" +
                                    "driver.get('http://example.com')" },
                            { question: "Selenium ile bir web elementine nasıl tıklanır?", answer: "click() fonksiyonu ile bir elemente tıklanır:\n" +
                                    "button = driver.find_element_by_id('submit')\n" +
                                    "button.click()\n" },
                            { question: "Selenium ile bir web elementinin metnini nasıl alırsınız?", answer: "text özelliği kullanılarak elementin metni alınabilir:\n" +
                                    "element = driver.find_element_by_tag_name('h1')\n" +
                                    "print(element.text)" },
                            { question: "Selenium ile bir sayfanın yüklenmesini nasıl bekleriz?", answer: "WebDriverWait ile belirli bir koşulun gerçekleşmesini bekleyebilirsiniz:\n" +
                                    "from selenium.webdriver.common.by import By\n" +
                                    "from selenium.webdriver.support.ui import WebDriverWait\n" +
                                    "from selenium.webdriver.support import expected_conditions as EC\n" +
                                    "\n" +
                                    "element = WebDriverWait(driver, 10).until(\n" +
                                    "    EC.presence_of_element_located((By.ID, 'element_id'))\n" +
                                    ")" },
                            { question: "Selenium ile bir açılır menüden (dropdown) nasıl seçim yapılır?", answer: "Select sınıfı kullanılarak açılır menüden seçim yapılabilir:\n" +
                                    "from selenium.webdriver.support.ui import Select\n" +
                                    "dropdown = driver.find_element_by_id('dropdown')\n" +
                                    "select = Select(dropdown)\n" +
                                    "select.select_by_visible_text('Option 1')" },
                            { question: "Selenium ile bir sayfa kaydırma işlemi nasıl yapılır?", answer: "execute_script() ile JavaScript kodu çalıştırılarak sayfa kaydırılabilir:\n" +
                                    "driver.execute_script(\"window.scrollTo(0, document.body.scrollHeight);\")" },
                            { question: "Selenium ile bir sayfada JavaScript alert penceresini nasıl kapatırsınız?", answer: "alert() fonksiyonu ile alert penceresi alınır ve accept() veya dismiss() metodu ile kapatılır:\n" +
                                    "alert = driver.switch_to.alert\n" +
                                    "alert.accept()  # Alert penceresini kabul et\n" +
                                    "# alert.dismiss()  # Alert penceresini reddet" },
                            { question: "Selenium ile bir sayfa üzerinde birden fazla sekme (tab) nasıl yönetilir?", answer: "window_handles ile sekmelerin tüm pencereleri alınabilir ve switch_to.window() ile istenen sekmeye geçilebilir:\n" +
                                    "main_window = driver.current_window_handle\n" +
                                    "driver.find_element_by_link_text('Open New Tab').click()\n" +
                                    "new_window = driver.window_handles[1]\n" +
                                    "driver.switch_to.window(new_window)\n" +
                                    "driver.close()  # Yeni sekmeyi kapat\n" +
                                    "driver.switch_to.window(main_window)  # Ana pencereye geri dön" },
                        ],
                    },
        ],
    };

    // Kategori ekle
    const result = await db.run("INSERT OR IGNORE INTO categories (name) VALUES (?)", [category.name]);
    const categoryId = result.lastID;

    // Konu ve soruları ekle
    for (const topic of category.topics) {
        const topicResult = await db.run("INSERT INTO topics (category_id, name) VALUES (?, ?)", [categoryId, topic.name]);
        const topicId = topicResult.lastID;

        for (const question of topic.questions) {
            await db.run("INSERT INTO questions (topic_id, question, answer) VALUES (?, ?, ?)", [
                topicId,
                question.question,
                question.answer,
            ]);
        }
    }

    console.log("Veriler başarıyla eklendi.");
}

// Çalıştırma
(async () => {
    await initializeDb();
    await insertData();
})();