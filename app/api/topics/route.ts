import { NextResponse } from "next/server";
import { getDb } from "../../../lib/db";

export async function POST(req: Request) {
    const { name, categoryId } = await req.json();
    const db = await getDb();

    await db.run("INSERT INTO topics (name, category_id) VALUES (?, ?)", name, categoryId);

    return NextResponse.json({ message: "Topic added" });
}

export async function GET(req: Request) {
    const url = new URL(req.url);
    const categoryId = url.searchParams.get("categoryId");
    const db = await getDb();

    const topics = await db.all("SELECT * FROM topics WHERE category_id = ?", categoryId);

    return NextResponse.json(topics);
}
