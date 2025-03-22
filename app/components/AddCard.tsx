'use client'; //React uygulamasının istemci tarafında çalışacağını belirtir.
import { useState } from 'react'; //Bileşen durumlarını yönetmek için kullanılır. Değerlerin değişmesi durumunda bileşenin yeniden render edilmesini sağlar.
import { Category, Topic } from '../types';


//aktarılan props türlerini tanımlar.
interface AddCardProps {
    onClose: () => void; //Card penceresini kapatmak için çağrılan bir fonksiyon.
    selectedCategory: Category | null; //Kullanıcının seçtiği kategori bilgilerini içerir
    selectedTopic: Topic | null; //Kullanıcının seçtiği konu bilgilerini içerir
    onAdd: () => void; //Yeni bir kategori, konu veya soru eklendikten sonra verileri yeniden çekmek için çağrılan bir fonksiyon.
}

export default function AddCard({ onClose, selectedCategory, selectedTopic, onAdd }: AddCardProps) {
    const [name, setName] = useState('');  //Kategori veya konu adını tutar.
    const [question, setQuestion] = useState(''); //Soru bilgisini tutar.
    const [answer, setAnswer] = useState(''); // Cevap bilgisini tutar.

    const handleAdd = async () => { //Yeni bir kategori, konu veya soru eklemek için kullanılan fonksiyon.
        if (selectedTopic) { //Eğer kullanıcı bir konu seçmişse
            await fetch('/api/questions', { //Yeni bir soru eklemek için /api/questions adresine POST isteği gönderilir.
                method: 'POST',
                body: JSON.stringify({ question, answer, topicId: selectedTopic.id }), //Yeni soru bilgileri gönderilir.
                headers: { 'Content-Type': 'application/json' }, //Gönderilen verinin JSON formatında olduğu belirtilir.
            });
        } else if (selectedCategory) { //Eğer kullanıcı bir kategori seçmişse
            await fetch('/api/topics', { //Yeni bir konu eklemek için /api/topics adresine POST isteği gönderilir.
                method: 'POST',
                body: JSON.stringify({ name, categoryId: selectedCategory.id }),
                headers: { 'Content-Type': 'application/json' },
            });
        } else { //Eğer kullanıcı bir kategori veya konu seçmemişse
            await fetch('/api/categories', { //Yeni bir kategori eklemek için /api/categories adresine POST isteği gönderilir. 
                method: 'POST',
                body: JSON.stringify({ name }),
                headers: { 'Content-Type': 'application/json' },
            });
        }
        onAdd(); // Verileri yeniden çekmek için onAdd fonksiyonu çağrılır.
        onClose(); // Card penceresi kapatılır.
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
            <div className="bg-white p-6 rounded shadow-lg">
                <h2 className="text-xl font-bold mb-4">Yeni Ekle</h2>
                {selectedTopic ? ( //Eğer kullanıcı bir konu seçmişse soru ve cevap bilgilerini alır.
                    <>
                        <input
                            type="text"
                            placeholder="Soru"
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            className="border p-2 w-full mb-4"
                        />
                        <input
                            type="text"
                            placeholder="Cevap"
                            value={answer}
                            onChange={(e) => setAnswer(e.target.value)}
                            className="border p-2 w-full mb-4"
                        />
                    </>
                ) : ( //Eğer kullanıcı bir konu seçmemişse kategori veya konu adını alır.
                    <input
                        type="text"
                        placeholder={selectedCategory ? 'Konu Adı' : 'Kategori Adı'}
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className="border p-2 w-full mb-4"
                    />
                )}
                <div className="flex justify-end gap-2">
                    <button onClick={onClose} className="px-4 py-2 bg-gray-300 rounded shadow">
                        İptal
                    </button>
                    <button onClick={handleAdd} className="px-4 py-2 bg-green-500 text-white rounded shadow">
                        Ekle
                    </button>
                </div>
            </div>
        </div>
    );
}
