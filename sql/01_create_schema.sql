-- FILE: sql/01_create_schema.sql
-- Creates database and 3-tier schema architecture
CREATE DATABASE credit_risk_db;
-- \c credit_risk_db
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS cleaned_data;
CREATE SCHEMA IF NOT EXISTS analytics;
