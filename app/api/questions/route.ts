import { NextResponse } from "next/server";
import { getDb } from "../../../lib/db";

export async function POST(req: Request) {
    const { question, answer, topicId } = await req.json();
    const db = await getDb();

    await db.run("INSERT INTO questions (question, answer, topic_id) VALUES (?, ?, ?)", question, answer, topicId);

    return NextResponse.json({ message: "Question added" });
}

export async function GET(req: Request) {
    const url = new URL(req.url);
    const topicId = url.searchParams.get("topicId");
    const db = await getDb();

    const questions = await db.all("SELECT * FROM questions WHERE topic_id = ?", topicId);

    return NextResponse.json(questions);
}
