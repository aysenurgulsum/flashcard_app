import sqlite3 from "sqlite3";
import { open, Database } from "sqlite";

/**
 * SQLite veritabanı bağlantısını açar.
 * Veritabanı, `database.sqlite` adında bir dosya olarak oluştu.
 */
export async function getDb(): Promise<Database> {
    return open({
        filename: "./database.sqlite", // Veritabanı dosyası adı
        driver: sqlite3.Database, // sqlite3 sürücüsü
    });
}
