# Audiobooks Frontend

This project is a frontend application for managing and streaming audiobooks. It provides a user interface for uploading ebooks, controlling audio playback, and interacting with an API for audio streaming.

## Project Structure

```
audiobooks-frontend
├── public
│   └── index.html          # Main HTML entry point
├── src
│   ├── frontend.js         # Main JavaScript code for user interactions and API calls
│   ├── components
│   │   └── player.js       # Audio player component
│   ├── styles
│   │   └── main.css        # CSS styles for the application
│   └── utils
│       └── api.js          # Utility functions for API calls
├── package.json             # npm configuration file
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd audiobooks-frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Usage

1. Start the development server:
   ```
   npm start
   ```

2. Open your browser and navigate to `http://localhost:8000` to access the application.

## Features

- Upload ebooks for processing.
- Stream audio playback with controls for play, pause, and stop.
- Ask questions via audio and receive audio responses.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.