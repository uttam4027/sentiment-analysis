# sentiment-analysis
Sentiment Analysis for jewelry reviews. Categorizes aspects such as Diamond, Style, Manufacturing, Merchandising, and Retailer, providing Positive, Negative, or Return sentiment and reasoning for each. Batch analyze reviews from Excel and export results easily.

# Sentiment Analysis for Jewelry Reviews

## Project Overview
This repository contains a sentiment analysis tool for jewelry product reviews. The tool identifies product aspects (e.g., Diamond, Style, Manufacturing, Merchandising, Retailer) and classifies sentiment as Positive, Negative, or Return.

## Features
- Categorizes reviews based on product aspects.
- Classifies sentiments at the aspect level.
- Extracts reasoning and keywords for sentiment.
- Supports review analysis via Excel input and outputs results to Excel.

## Requirements
- Python 3.x
- pandas, numpy, tqdm, requests, re, json, openpyxl
- langchain, sentence-transformers, faiss
- Excel file with columns: Product Style, Timeline, Date, Review

## Usage

1. Clone this repository:

