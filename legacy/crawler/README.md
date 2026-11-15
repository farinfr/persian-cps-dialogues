# Digikala Product Seeder

A Python service that fetches product data from the Digikala API and stores it in a PostgreSQL database. This service handles product listings, detailed product information, and category data.

## Features

- ✅ Fetches product listings from Digikala categories
- ✅ Retrieves detailed product information
- ✅ Stores products and categories in separate PostgreSQL tables
- ✅ Filters and saves only meaningful product specifications
- ✅ Handles pagination automatically
- ✅ Rate limiting to avoid API throttling
- ✅ Update existing products or create new ones (upsert functionality)
- ✅ Comprehensive logging

## Database Schema

### Categories Table
- `id` - Primary key
- `digikala_id` - Unique Digikala category ID
- `title` - Category title (Persian)
- `title_en` - Category title (English)
- `code` - Category code
- `url` - Category URL path
- `created_at` - Creation timestamp
- `updated_at` - Last update timestamp

### Products Table
- `id` - Primary key
- `digikala_id` - Unique Digikala product ID
- `category_id` - Foreign key to categories table
- `title` - Product title (Persian)
- `title_en` - Product title (English)
- `url` - Product URL
- `brand` - Brand name
- `price` - Original price
- `selling_price` - Current selling price
- `discount_percent` - Discount percentage
- `status` - Product status
- `is_incredible` - Incredible offer flag
- `is_fresh` - Fresh product flag
- `main_image_url` - Main product image URL
- `images` - JSON array of all product images
- `rating_average` - Average rating
- `rating_count` - Number of ratings
- `is_available` - Availability status
- `specifications` - JSON object with filtered product specs
- `created_at` - Creation timestamp
- `updated_at` - Last update timestamp

## Setup

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher
- pip (Python package manager)

### Installation

1. Clone or download this project to your local machine

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Create a PostgreSQL database:
```bash
createdb digikala_products
```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Update the database credentials and configuration:

```bash
cp .env.example .env
```

Edit `.env` file:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=digikala_products
DB_USER=your_postgres_username
DB_PASSWORD=your_postgres_password

DIGIKALA_BASE_URL=https://api.digikala.com
API_REQUEST_TIMEOUT=30
```

5. Initialize the database tables:
```bash
python main.py --init-db
```

## Usage

### Basic Usage

Seed products from the default category (mobile-phone):
```bash
python main.py
```

### Advanced Options

**Seed a specific category:**
```bash
python main.py --category laptop
```

**Limit the number of pages:**
```bash
python main.py --category mobile-phone --max-pages 5
```

**Skip fetching detailed product information (faster but less data):**
```bash
python main.py --no-details
```

**Adjust delay between requests (to avoid rate limiting):**
```bash
python main.py --delay 1.0
```

**Initialize database tables:**
```bash
python main.py --init-db
```

**Combine multiple options:**
```bash
python main.py --category mobile-phone --max-pages 10 --delay 0.8
```

### Command Line Arguments

- `--category` - Category slug to seed (default: mobile-phone)
- `--max-pages` - Maximum number of pages to fetch (default: all pages)
- `--no-details` - Skip fetching detailed product information
- `--delay` - Delay between API requests in seconds (default: 0.5)
- `--init-db` - Initialize database tables

## API Endpoints Used

### List Products API
```
GET https://api.digikala.com/v1/categories/{category}/search/?page={page}&_rch={rch}
```

Example: `https://api.digikala.com/v1/categories/mobile-phone/search/?page=1&_rch=db340a7f7c4f`

### Product Detail API
```
GET https://api.digikala.com/v2/product/{product_id}/?variant_id={variant_id}&_rch={rch}
```

Example: `https://api.digikala.com/v2/product/16552148/?variant_id=72423124&_rch=db340a7f7c4f`

## Project Structure

```
seeder-app/
├── main.py              # Main entry point
├── config.py            # Configuration management
├── models.py            # SQLAlchemy database models
├── database.py          # Database connection and management
├── api_client.py        # Digikala API client
├── seeder.py            # Product seeding service
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment configuration
└── README.md           # This file
```

## How It Works

1. **Connection**: Connects to PostgreSQL database using SQLAlchemy
2. **Category Fetching**: Fetches the category information from the first page
3. **Product Listing**: Retrieves all products from the category (handles pagination)
4. **Product Details**: For each product, fetches detailed information (optional)
5. **Data Filtering**: Filters specifications to keep only meaningful data
6. **Database Storage**: Saves/updates categories and products in the database
7. **Logging**: Provides detailed logging throughout the process

## Meaningful Specifications

The seeder filters product specifications to save only meaningful data. The following specification types are preserved:

- Memory & Storage (RAM, ROM, Storage)
- Screen (Screen Size, Display, Resolution, Refresh Rate)
- Processor (CPU, Chipset, GPU)
- Camera specifications
- Battery information
- Physical properties (Weight, Dimensions)
- Operating System
- Connectivity & Network
- Color & Material
- Warranty information

## Error Handling

- Database connection failures are logged and the program exits gracefully
- API request failures are logged but don't stop the seeding process
- Products that fail to save are logged individually
- Keyboard interruption (Ctrl+C) is handled gracefully

## Performance Considerations

- **Rate Limiting**: Configurable delay between API requests to avoid throttling
- **Batch Processing**: Processes multiple products efficiently
- **Database Optimization**: Uses indexes on key fields for faster queries
- **Upsert Logic**: Updates existing records instead of creating duplicates

## Logging

The application provides detailed logging at multiple levels:
- INFO: General progress and successful operations
- WARNING: Unexpected but handled situations
- ERROR: Failed operations with details

Logs include timestamps and are formatted for easy reading.

## Troubleshooting

**Database connection error:**
- Verify PostgreSQL is running
- Check credentials in `.env` file
- Ensure database exists

**API request failures:**
- Check internet connection
- Verify API endpoints are accessible
- Try increasing `--delay` parameter

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## License

This project is provided as-is for educational and personal use.

## Contributing

Feel free to submit issues or pull requests for improvements.

## Author

Created for seeding Digikala product data into PostgreSQL database.

