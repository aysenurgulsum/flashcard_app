export type Category = {
    id: number;
    name: string;
    topics: Topic[];
};

export type Topic = {
    id: number;
    category_id: number;
    name: string;
    questions?: Question[];
};

export type Question = {
    id: number;
    topic_id: number;
    question: string;
    answer: string;
};
