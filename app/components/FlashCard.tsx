interface FlashCardProps { // Aktarılan props'lar
    topic: string; 
    onClick: () => void; 
    isSelected?: boolean; 
}

export default function FlashCard({ topic, onClick, isSelected }: FlashCardProps) {
    return (
        <div
            onClick={onClick} // Butona  tıklandıysa
            className={`cursor-pointer rounded-lg shadow-lg p-4 text-center text-lg font-bold text-white transition ${
                isSelected // Seçili olma durumuna göre renk değişimi
                    ? "bg-gradient-to-r from-teal-400 to-teal-600"
                    : "bg-gradient-to-r from-purple-500 to-purple-900 hover:scale-105"
            }`}
        >
            {topic}
        </div>
    );
}
