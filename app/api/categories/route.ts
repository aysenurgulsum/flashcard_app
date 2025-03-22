import { NextResponse } from "next/server"; // Next.js sunucu yanıtlarını içe aktar
import { getDb } from "../../../lib/db"; // Veritabanı bağlantısı db.ts dosyasında

export async function POST(req: Request) {
    const { name } = await req.json(); // İstekten kategori adını al
    const db = await getDb();

    await db.run("INSERT INTO categories (name) VALUES (?)", name); // Kategori ekle

    return NextResponse.json({ message: "Kategori eklendi." });
}

export async function GET() {
    const db = await getDb();
    const categories = await db.all("SELECT * FROM categories"); // Tüm kategorileri al

    return NextResponse.json(categories);
}
