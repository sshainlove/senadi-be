-- Create database if not exists
CREATE DATABASE IF NOT EXISTS `chatbot`;
USE `chatbot`;

-- Users table
CREATE TABLE IF NOT EXISTS `users` (
    `id` VARCHAR(36) PRIMARY KEY,
    `username` VARCHAR(255) NOT NULL,
    `email` VARCHAR(255) UNIQUE NOT NULL,
    `password` VARCHAR(255) NOT NULL,
    `is_admin` BOOLEAN DEFAULT FALSE
);

-- Chats table
CREATE TABLE IF NOT EXISTS `chats` (
    `id` VARCHAR(36) PRIMARY KEY,
    `user_id` VARCHAR(36) NOT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `title` VARCHAR(255),
    `verified` BOOLEAN DEFAULT FALSE,
    `verified_at` DATETIME,
    `verified_by` VARCHAR(36),
    FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE,
    FOREIGN KEY (`verified_by`) REFERENCES `users`(`id`) ON DELETE SET NULL
);

-- Messages table
CREATE TABLE IF NOT EXISTS `messages` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `chat_id` VARCHAR(36) NOT NULL,
    `sender` VARCHAR(10) NOT NULL,
    `message` TEXT NOT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `is_corrected` BOOLEAN DEFAULT FALSE,
    `is_correction` BOOLEAN DEFAULT FALSE,
    `corrected_message_id` INT,
    `corrected_by` VARCHAR(36),
    FOREIGN KEY (`chat_id`) REFERENCES `chats`(`id`) ON DELETE CASCADE,
    FOREIGN KEY (`corrected_by`) REFERENCES `users`(`id`) ON DELETE SET NULL,
    FOREIGN KEY (`corrected_message_id`) REFERENCES `messages`(`id`) ON DELETE SET NULL
);

-- Knowledge files table
CREATE TABLE IF NOT EXISTS `knowledge_files` (
    `id` VARCHAR(36) PRIMARY KEY,
    `filename` VARCHAR(255) NOT NULL,
    `file_type` VARCHAR(50) NOT NULL,
    `content` LONGTEXT,
    `upload_date` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `uploaded_by` VARCHAR(36) NOT NULL,
    FOREIGN KEY (`uploaded_by`) REFERENCES `users`(`id`) ON DELETE CASCADE
);

-- Insert default admin user
INSERT INTO `users` (`id`, `username`, `email`, `password`, `is_admin`) 
VALUES (UUID(), 'admin', 'admin@example.com', '$2b$12$1tJhwzsFxHJB1kYpZFOF8O6jGQIJGXg8WGLnHRW7MiLfts9ZySyyC', TRUE)
ON DUPLICATE KEY UPDATE `email` = 'admin@example.com';
-- Default password is 'admin123' (hashed with bcrypt) 