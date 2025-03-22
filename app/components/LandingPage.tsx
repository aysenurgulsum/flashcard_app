interface LandingPageProps {
    onGetStarted: () => void; // Anasayfadaki butona  tıklanınca çağrılacak fonksiyon
}

export default function LandingPage({ onGetStarted }: LandingPageProps) {
    return (
        <div className="h-screen flex flex-col bg-black text-white">
            {/* Üst Menü */}
            <header className="bg-gradient-to-r from-purple-500 to-pink-300 text-white py-4 shadow-lg">
                <div className="max-w-7xl px-4 flex justify-start items-center">
                    <h1 className="text-3xl font-extrabold tracking-wide italic">FlashLearn</h1>
                </div>
            </header>

            {/* İçerik */}
            <main className="flex flex-1 justify-center items-center text-center p-4">
                <div className="flex flex-row justify-between items-center w-full max-w-7xl">
                    {/* Sol Kısım (Yazılar ve Buton) */}
                    <div className="flex flex-col items-start">
                        <h2 className="text-4xl font-extrabold mb-4">Öğren, Tekrarla, Başar!</h2>
                        <p className="text-gray-400 text-lg mb-6">
                            Bilginizi artırmak ve hafızanızı güçlendirmek için etkili bir yöntem.
                            <br/>
                            Kendi öğrenme ihtiyaçlarınıza uygun flash kartlar oluşturun ve kolayca gözden geçirin.
                        </p>
                        <button
                            onClick={onGetStarted} // Props üzerinden gelen fonksiyonu çağır
                            className="px-6 py-3 bg-purple-600 text-white font-semibold rounded-md shadow-lg hover:bg-purple-700 transition"
                        >
                            Hemen Başlayın
                        </button>
                    </div>

                    {/* Görsel bölümü */}
                    <aside className="mt-8">
                        <img
                            src="/images/FlashCard5.png"
                            alt="Students Illustration"
                            className="w-full max-w-2xl mx-auto" // max-w-2xl ile daha büyük görsel boyutu
                        />
                    </aside>
                </div>
            </main>
        </div>
    );
}
