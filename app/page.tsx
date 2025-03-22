'use client';
import { useState, useEffect } from 'react';
import FlashCard from './components/FlashCard';
import QuestionPanel from './components/QuestionPanel';
import AddCard from './components/AddCard';
import LandingPage from './components/LandingPage'
import { Category, Topic, Question } from './types';

export default function Home() {
    const [showLandingPage, setShowLandingPage] = useState(true);  // LandingPage'i göstermek için
    const [categories, setCategories] = useState<Category[]>([]); // Kategorileri tutmak için
    const [topics, setTopics] = useState<Topic[]>([]); // Konuları tutmak için
    const [selectedCategory, setSelectedCategory] = useState<Category | null>(null); // Seçilen kategoriyi tutmak için
    const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null); // Seçilen konuyu tutmak için
    const [showAddCard, setShowAddCard] = useState(false);  // AddCard'ı göstermek için

    // Kategorileri, konuları ve soruları çekmek için
    const fetchCategories = async () => {
        try {
            const res = await fetch('/api/categories');
            if (!res.ok) {
                throw new Error('Failed to fetch categories');
            }
            const data: Category[] = await res.json();
            setCategories(data);
        } catch (error) {
            console.error('Error fetching categories:', error);
        }
    };

    const fetchTopics = async () => {
        if (selectedCategory) {
            try {
                const res = await fetch(`/api/topics?categoryId=${selectedCategory.id}`);
                if (!res.ok) {
                    throw new Error('Failed to fetch topics');
                }
                const data: Topic[] = await res.json();
                setTopics(data);
            } catch (error) {
                console.error('Error fetching topics:', error);
            }
        }
    };

    const fetchQuestions = async () => {
        if (selectedTopic) {
            try {
                const res = await fetch(`/api/questions?topicId=${selectedTopic.id}`);
                if (!res.ok) {
                    throw new Error('Failed to fetch questions');
                }
                const data: Question[] = await res.json();
                setSelectedTopic((prevTopic) => prevTopic ? { // Eğer selectedTopic varsa güncelle 
                    ...prevTopic, //  Eğer prevTopic (önceki konu) tanımlıysa, ...prevTopic ile mevcut özellikleri korunur.
                    questions: data, //Gelen questions (sorular), prevTopic'in içine eklenir.
                } : null);
            } catch (error) {
                console.error('Error fetching questions:', error);
            }
        }
    };

    // Kategorileri, konuları ve soruları çekmek için useEffect kullanılır
    useEffect(() => {  
        fetchCategories();
    }, []); // []:  useEffect sadece ilk renderda çalışır

    useEffect(() => {
        fetchTopics();
    }, [selectedCategory]); // selectedCategory değiştiğinde çalışır

    useEffect(() => {
        fetchQuestions();
    }, [selectedTopic]); // selectedTopic değiştiğinde çalışır

    const handleCategoryClick = (categoryId: number) => {
        const category = categories.find((c) => c.id === categoryId);
        setSelectedCategory(category || null);
        setSelectedTopic(null); // Reset selected topic when category changes
    };

    const handleTopicClick = (topicId: number) => { // Seçilen konuyu almak için
        const topic = topics.find((t) => t.id === topicId);
        setSelectedTopic(topic ? { ...topic, questions: topic.questions || [] } : null); // Seçilen konuyu güncelle
    };

    const handleBack = () => { // Geri gitmek için
        if (selectedTopic) {
            setSelectedTopic(null);
        } else {
            setSelectedCategory(null);
        }
    };
 
    const openAddCard = () => setShowAddCard(true); // AddCard'ı açmak için 
    const closeAddCard = () => setShowAddCard(false); // AddCard'ı kapatmak için

    const handleAdd = () => {  //Verileri tekrar güncelleme
        fetchCategories();
        fetchTopics();
        fetchQuestions();
    };
        // Eğer Landing Page gösteriliyorsa
        if (showLandingPage) {
            return <LandingPage onGetStarted={() => setShowLandingPage(false)} />; // LandingPage'i çağır
        }

    return (
        <div className="flex flex-col h-screen">
            <header className="bg-gradient-to-r from-purple-500 to-pink-300 text-white py-4 shadow-lg">
                <div className="max-w-7xl px-4 flex justify-start items-center">
                    <h1
                        className="text-3xl font-extrabold tracking-wide italic cursor-pointer"
                        onClick={() => setShowLandingPage(true)} // FlashLearn yazısına tıklandığında LandingPage'e git
                    >
                        FlashLearn
                    </h1>
                </div>
            </header>

            {selectedCategory && !selectedTopic && ( // Eğer seçilen kategori varsa ve seçilen konu yoksa
                <button
                    onClick={handleBack}
                    className="absolute top-20 left-4 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-600 hover:to-pink-600 rounded shadow z-10"
                >
                    Ana Menü
                </button>
            )}

            <div className="flex flex-1">
                {!selectedCategory ? ( // Eğer seçilen kategori yoksa 
                    <div className="grid grid-cols-3 gap-4 p-8 w-full">
                        {categories.map((category) => ( // Kategorileri göster
                            <FlashCard
                                key={category.id}
                                topic={category.name}
                                onClick={() => handleCategoryClick(category.id)}
                            />
                        ))}
                        <button
                            onClick={openAddCard} // Kategori eklemek için AddCard'ı aç
                            className="bg-gradient-to-br from-cyan-500 to-pink-400 text-white font-bold p-4 rounded shadow hover:bg-green-600"
                        >
                            Kategori Ekle
                        </button>
                    </div>
                ) : (
                    <div className="flex flex-1">
                        {selectedTopic && ( // Eğer seçilen konu varsa
                            <div className="w-1/4 bg-gradient-to-b from-purple-200 via-purple-300 to-purple-400 p-4">
                                {topics.map((topic) => ( // Konuları göster, sol kısım
                                    <FlashCard
                                        key={topic.id}
                                        topic={topic.name}
                                        onClick={() => handleTopicClick(topic.id)}
                                        isSelected={topic.id === selectedTopic.id}
                                    />
                                ))}
                            </div>
                        )}
                        <div className={selectedTopic ? 'w-3/4 p-4' : 'w-full p-8'}> 
                            {!selectedTopic ? ( // Eğer seçilen konu yoksa
                                <div className="grid grid-cols-3 gap-4 p-8 w-full">
                                    {topics.map((topic) => ( // Konuları göster
                                        <FlashCard
                                            key={topic.id}
                                            topic={topic.name}
                                            onClick={() => handleTopicClick(topic.id)}
                                        />
                                    ))}
                                    <button
                                        onClick={openAddCard} // Konu eklemek için AddCard'ı aç
                                        className="bg-gradient-to-br from-cyan-500 to-pink-400 text-white font-bold p-4 rounded shadow hover:bg-green-600 mt-4"
                                    >
                                        Konu Ekle
                                    </button>
                                </div>
                            ) : ( // Eğer seçilen konu varsa
                                <>
                                    <QuestionPanel
                                        questions={selectedTopic.questions || []} // Soruları göster
                                        onBack={handleBack}     // Geri gitmek için
                                    />
                                    <button
                                        onClick={openAddCard} // Soru eklemek için AddCard'ı aç
                                        className="bg-gradient-to-br from-cyan-500 to-pink-400 text-white font-bold p-4 rounded shadow hover:bg-green-600 mt-4"
                                    >
                                        Soru Ekle
                                    </button>
                                </>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {showAddCard && (
                <AddCard
                    onClose={closeAddCard} // AddCard'ı kapat
                    selectedCategory={selectedCategory} // Seçilen kategoriyi gönder
                    selectedTopic={selectedTopic} // Seçilen konuyu gönder
                    onAdd={handleAdd} // Verileri güncelle
                />
            )}
        </div>
    );
}
