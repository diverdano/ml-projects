--DROP SCHEMA IF EXISTS flights_dev;
--CREATE SCHEMA flights_dev;

--DROP GROUP flights_dev_rw;
--CREATE GROUP flights_dev_rw;

--GRANT USAGE ON SCHEMA flights_dev TO GROUP flights_dev_rw;
--GRANT SELECT, UPDATE, DELETE, INSERT ON ALL TABLES IN SCHEMA flights_dev TO GROUP flights_dev_rw;
--GRANT CREATE ON SCHEMA flights_dev TO GROUP flights_dev_rw;

--DROP USER IF EXISTS dev_etl;
--CREATE USER sor_etl WITH PASSWORD 'Passw0rd' IN GROUP flights_dev_rw;

show timezone;
set timezone = 'America/Los_Angeles';

create table flight_log(
    'id_trans' int not null,
    'date_flight' timestamptz not null,
    'id_airline' varchar(2) not null,
    'id_tailnum' varchar(6),
    'id_flightnum' int not null,
    'id_airport_orig' varchar(3),
    'id_airport_dest' varchar(3),
    'time_depart_crs' timestamptz not null,
    'time_depart' timestamptz,
    'time_depart_delay' interval,
    'time_taxi_out' int,
    'time_wheelsoff' int,
    'time_wheelson' int,
    'time_taxi_in' int,
    'time_arrive_crs' timestamptz not null,
    'time_arrive' timestamptz,
    'time_arrive_delay' interval,
    'time_elapsed_crs' interval,
    'time_elapsed_act' interval,
    'stat_cancelled' int,
    'stat_diverted' int,
    'stat_miles' int          -- keep imperial measure
);

create table airlines (
    id_airline varchar(2) not null,
    name varchar(30) not null,
    primary key (id_airline)
);

create table airports(
    id_airport varchar(3) not null,
    name varchar(55) not null,
    city varchar(30) not null,
    st varchar(2) not null,
    state varchar(50),                 -- (iso standard, useful?)
    primary key (id_airport)
);

create table airplanes(
    id_tailnum varchar(6) not null,
    id_airline varchar(2) not null,   -- (could be transferred)
    id_tailnum_err varchar(6)         -- to hold incorrect tail numbers from client source dataset
--    date_service_first
--    date_service_last
--    cum_miles (calc from all segments)
);

create table sources(
    id integer not null,
    name varchar(50) not null,
    url varchar(255) not null,
    note varchar(255) not null
)

CREATE TABLE asset (
	id INTEGER NOT NULL,
	type VARCHAR(10) NOT NULL,
	symbol VARCHAR(25) NOT NULL,
	description VARCHAR(25) NOT NULL,
	exchange VARCHAR(10) NOT NULL,
	size INTEGER,
  coupon FLOAT,
  expiry VARCHAR(25),
  strike INTEGER,
	PRIMARY KEY (id)
);
CREATE TABLE allocation (
	id INTEGER NOT NULL,
  date_mod VARCHAR(25) NOT NULL,
	portfolio VARCHAR(10) NOT NULL,
	asset_id INTEGER,
	allocation FLOAT NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(asset_id) REFERENCES asset (id)
);

INSERT INTO asset (type, symbol, description, exchange, size) Values
  ("equity", "AMZN", "Amazon.com, Inc.", "NASDAQ", 1),
  ("equity", "GOOG", "Alphabet, Inc.", "NASDAQ", 1),
  ("future", "VIX", "CBOE Volatility Index", "CBOE", 1000),
  ("future", "ES", "S&P500 e-mini", "CME", 50),
  ("option", "AMZN 171215C01170000", "Amazon.com, Inc. Dec 15 2017 1170 Call", "CBO", 100),
  ("equity", "AAPL", "Apple, Inc.", "NASDAQ", 1),
  ("equity", "IBM", "International Business Machines, Inc.", "NYSE", 1),
  ("bond", "912828U57", "US Treasury Note 2.125 30nov2023 7Y", "OTC", 1000);

INSERT INTO allocation (date_mod, portfolio, asset_id, allocation) Values
  ("2017-06-01T23:00:00Z", "port1", 1, 0.25),
  ("2017-06-01T23:00:00Z", "port1", 3, 0.25),
  ("2017-06-01T23:00:00Z", "port1", 5, 0.25),
  ("2017-06-01T23:00:00Z", "port1", 8, 0.25),
  ("2017-07-01T23:00:00Z", "port1", 1, 0.20),
  ("2017-07-01T23:00:00Z", "port1", 3, 0.20),
  ("2017-07-01T23:00:00Z", "port1", 5, 0.20),
  ("2017-07-01T23:00:00Z", "port1", 8, 0.20),
  ("2017-08-01T23:00:00Z", "port1", 1, 0.15),
  ("2017-08-01T23:00:00Z", "port1", 3, 0.15),
  ("2017-08-01T23:00:00Z", "port1", 5, 0.15),
  ("2017-08-01T23:00:00Z", "port1", 8, 0.15),
  ("2017-09-01T23:00:00Z", "port1", 1, 0.30),
  ("2017-09-01T23:00:00Z", "port1", 3, 0.30),
  ("2017-09-01T23:00:00Z", "port1", 5, 0.30);
