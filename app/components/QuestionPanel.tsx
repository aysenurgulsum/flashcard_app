import { useState } from "react"; 

interface Question { 
    question: string;
    answer: string;
}

interface QuestionPanelProps {
    questions: Question[];
    onBack: () => void;
}

export default function QuestionPanel({ questions, onBack }: QuestionPanelProps) {
    const [currentIndex, setCurrentIndex] = useState(0); // Soru indexini tutacak state
    const [showAnswer, setShowAnswer] = useState(false); // Cevabı göster/gizle state

    const handleNext = () => { // Sonraki soruya geçmek için
        setShowAnswer(false);
        setCurrentIndex((prev) => (prev + 1) % questions.length);
    };

    const handlePrev = () => { // Önceki soruya geçmek için
        setShowAnswer(false);
        setCurrentIndex((prev) => (prev - 1 + questions.length) % questions.length);
    };

    if (questions.length === 0) { // Eğer soru yoksa
        return (
            <div className="p-4">
                <button
                    onClick={onBack}
                    className="mb-4 px-4 py-2 bg-gradient-to-r from-purple-600 to-purple-400 text-white hover:from-purple-700 hover:to-purple-500 rounded"
                >
                    Geri Dön
                </button>
                <div className="border p-4 rounded shadow">
                    <h2 className="text-lg font-bold">Hiç soru bulunamadı.</h2>
                </div>
            </div>
        );
    }

    return (
        <div className="p-4">
            <button
                onClick={onBack} // Geri dön butonuna tıklanınca onBack fonksiyonunu çalışt
                className="mb-4 px-4 py-2 bg-gradient-to-r from-purple-600 to-purple-400 text-white hover:from-purple-700 hover:to-purple-500 rounded"
            >
                Geri Dön
            </button>
            <div className="border p-4 rounded shadow">
                <h2 className="text-lg font-bold">{questions[currentIndex].question}</h2>
                {showAnswer && <p className="mt-2 text-gray-700">{questions[currentIndex].answer}</p>}
                <button
                    onClick={() => setShowAnswer(!showAnswer)} // Cevabı göster/gizle butonuna tıklanınca showAnswer state'ini tersine çevir
                    className="mt-4 px-4 py-2 bg-gradient-to-r from-teal-400 to-teal-600 text-white hover:from-teal-500 hover:to-teal-700 rounded"
                >
                    {showAnswer ? "Cevabı Gizle" : "Cevabı Göster"} 
                </button>
            </div>
            <div className="flex justify-between mt-4">
                <button
                    onClick={handlePrev}
                    className="mt-4 px-4 py-2 bg-gradient-to-r from-purple-600 to-purple-400 text-white hover:from-purple-700 hover:to-purple-500 rounded"
                >
                    Önceki Soru
                </button>
                <button
                    onClick={handleNext}
                    className="mt-4 px-4 py-2 bg-gradient-to-r from-purple-600 to-purple-400 text-white hover:from-purple-700 hover:to-purple-500 rounded"
                >
                    Sonraki Soru
                </button>
            </div>
        </div>
    );
}