# GSTVision AI

**Turn scanned GST 2A statements and E-Way Bills into clean Excel reports — automatically.**

## What this is

GSTVision AI is a document automation tool for Indian GST reconciliation. You give it a GST 2A statement or an E-Way Bill — as a phone photo, a scanned copy, or a digital PDF — and it reads the document, pulls out every field (GSTIN, invoice number, invoice value, IGST/CGST/SGST, and more), and hands you back a ready-to-use Excel sheet.

## Why it's useful

Every GST-registered business has to reconcile these documents against its own books every filing period. In real life, that means dozens or hundreds of scanned, skewed, and inconsistently formatted documents that someone has to manually re-type into a spreadsheet — slow work where a single mistyped GSTIN can throw off an entire reconciliation.

GSTVision AI automates that: upload a document (or a whole batch), and get back structured data in minutes instead of hours. It also gets smarter over time — every correction a user makes to a wrong field is remembered and used to improve future extractions.

## Features

- Upload one file or many at once — PDFs, JPGs, PNGs, scans, phone photos
- Works on digital PDFs and photographed/scanned documents alike
- Automatically straightens, cleans up, and reads poor-quality scans
- Exports a clean, ready-to-use Excel report for every job
- Learns from corrections and improves itself over time

## How to run it on your own computer

You'll need these installed first:
- **Python 3.10+**
- **Node.js 18+**
- **PostgreSQL** (running locally)

There are three parts to start, each in its own terminal window.

### 1. Get the project and set it up

```bash
git clone https://github.com/Arjunkalliyadath/GST2A-Document-AI.git
cd GST2A-Document-AI
cp .env.example .env
```
Open `.env` and fill in your PostgreSQL username/password.

### 2. Start the API (Terminal 1)

```bash
cd backend/django_app
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
This runs at **http://127.0.0.1:8000**

### 3. Start the AI engine (Terminal 2)

```bash
cd backend/fastapi_service
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
This runs at **http://127.0.0.1:8001**

> This step installs the OCR and AI libraries, so it takes a while and needs a good amount of free disk space (10+ GB). No GPU needed — it runs fine on a regular laptop, just a bit slower.

### 4. Start the website (Terminal 3)

```bash
cd frontend
npm install
npm run dev
```
This runs at **http://127.0.0.1:5173** — open that link in your browser to use the app.

That's it — with all three running, you can upload a document from the dashboard and download the Excel report once it's processed.

## Author

**Arjun K**
- GitHub: [@Arjunkalliyadath](https://github.com/Arjunkalliyadath)
- Email: arjunkalliyadath2001@gmail.com

Built and maintained by me as a practical, real-world application of document AI to GST compliance work in India.
