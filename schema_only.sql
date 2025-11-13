--
-- PostgreSQL database dump
--

\restrict EPsTYz57IKv4gNtaicsI8O1c8r9GLHpSV2Qtvvg65MR6db9sL31C5ekcuGZzEeg

-- Dumped from database version 18.0 (Homebrew)
-- Dumped by pg_dump version 18.0 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: consumption; Type: TABLE; Schema: public; Owner: meghanarendrasimha
--

CREATE TABLE public.consumption (
    transaction_id text NOT NULL,
    date date NOT NULL,
    inventory_id text,
    quantity_consumed integer,
    department text,
    staff_id text,
    shift text,
    consumption_reason text,
    remaining_stock integer,
    batch_lot text,
    CONSTRAINT consumption_shift_check CHECK ((shift = ANY (ARRAY['Morning'::text, 'Afternoon'::text, 'Night'::text])))
);


ALTER TABLE public.consumption OWNER TO meghanarendrasimha;

--
-- Name: finance; Type: TABLE; Schema: public; Owner: meghanarendrasimha
--

CREATE TABLE public.finance (
    invoice_id text NOT NULL,
    vendor_id text,
    inventory_id text,
    purchase_date date,
    quantity integer,
    unit_cost numeric,
    total_cost numeric,
    payment_status text,
    account_code text,
    delivery_date date,
    CONSTRAINT finance_payment_status_check CHECK ((payment_status = ANY (ARRAY['Paid'::text, 'Pending'::text, 'Overdue'::text])))
);


ALTER TABLE public.finance OWNER TO meghanarendrasimha;

--
-- Name: inventory_daily; Type: TABLE; Schema: public; Owner: meghanarendrasimha
--

CREATE TABLE public.inventory_daily (
    record_id integer NOT NULL,
    date date NOT NULL,
    inventory_id text,
    item_name text,
    opening_stock integer,
    quantity_consumed integer,
    quantity_restocked integer,
    closing_stock integer,
    vendor_id text,
    lead_time_days integer,
    department_count integer
);


ALTER TABLE public.inventory_daily OWNER TO meghanarendrasimha;

--
-- Name: inventory_daily_record_id_seq; Type: SEQUENCE; Schema: public; Owner: meghanarendrasimha
--

CREATE SEQUENCE public.inventory_daily_record_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.inventory_daily_record_id_seq OWNER TO meghanarendrasimha;

--
-- Name: inventory_daily_record_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: meghanarendrasimha
--

ALTER SEQUENCE public.inventory_daily_record_id_seq OWNED BY public.inventory_daily.record_id;


--
-- Name: inventory_department_mapping; Type: TABLE; Schema: public; Owner: meghanarendrasimha
--

CREATE TABLE public.inventory_department_mapping (
    inventory_id text,
    vendor_id text,
    department_code text,
    department_name text,
    lead_time_days integer,
    min_stock_limit integer,
    max_capacity integer,
    team_member text,
    team_member_email text,
    manager text,
    manager_email text
);


ALTER TABLE public.inventory_department_mapping OWNER TO meghanarendrasimha;

--
-- Name: inventory_master; Type: TABLE; Schema: public; Owner: meghanarendrasimha
--

CREATE TABLE public.inventory_master (
    inventory_id text NOT NULL,
    item_type text,
    item_name text NOT NULL,
    vendor_id text,
    lead_time_days integer,
    avg_daily_consumption numeric,
    minimum_required integer,
    maximum_capacity integer,
    initial_stock integer,
    unit_cost numeric,
    expiry_date date,
    embedding public.vector(384),
    CONSTRAINT inventory_master_item_type_check CHECK ((item_type = ANY (ARRAY['Medication'::text, 'Consumable'::text, 'Equipment'::text])))
);


ALTER TABLE public.inventory_master OWNER TO meghanarendrasimha;

--
-- Name: vendor_master; Type: TABLE; Schema: public; Owner: meghanarendrasimha
--

CREATE TABLE public.vendor_master (
    vendor_id text NOT NULL,
    vendor_name text NOT NULL,
    contact_number text,
    default_lead_time_days integer,
    region text,
    vendor_rating numeric,
    CONSTRAINT vendor_master_vendor_rating_check CHECK (((vendor_rating >= (1)::numeric) AND (vendor_rating <= (5)::numeric)))
);


ALTER TABLE public.vendor_master OWNER TO meghanarendrasimha;

--
-- Name: inventory_daily record_id; Type: DEFAULT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.inventory_daily ALTER COLUMN record_id SET DEFAULT nextval('public.inventory_daily_record_id_seq'::regclass);


--
-- Name: consumption consumption_pkey; Type: CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.consumption
    ADD CONSTRAINT consumption_pkey PRIMARY KEY (transaction_id);


--
-- Name: finance finance_pkey; Type: CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.finance
    ADD CONSTRAINT finance_pkey PRIMARY KEY (invoice_id);


--
-- Name: inventory_daily inventory_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.inventory_daily
    ADD CONSTRAINT inventory_daily_pkey PRIMARY KEY (record_id);


--
-- Name: inventory_master inventory_master_pkey; Type: CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.inventory_master
    ADD CONSTRAINT inventory_master_pkey PRIMARY KEY (inventory_id);


--
-- Name: vendor_master vendor_master_pkey; Type: CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.vendor_master
    ADD CONSTRAINT vendor_master_pkey PRIMARY KEY (vendor_id);


--
-- Name: consumption consumption_inventory_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.consumption
    ADD CONSTRAINT consumption_inventory_id_fkey FOREIGN KEY (inventory_id) REFERENCES public.inventory_master(inventory_id);


--
-- Name: finance finance_inventory_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.finance
    ADD CONSTRAINT finance_inventory_id_fkey FOREIGN KEY (inventory_id) REFERENCES public.inventory_master(inventory_id);


--
-- Name: finance finance_vendor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.finance
    ADD CONSTRAINT finance_vendor_id_fkey FOREIGN KEY (vendor_id) REFERENCES public.vendor_master(vendor_id);


--
-- Name: inventory_daily inventory_daily_inventory_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.inventory_daily
    ADD CONSTRAINT inventory_daily_inventory_id_fkey FOREIGN KEY (inventory_id) REFERENCES public.inventory_master(inventory_id);


--
-- Name: inventory_daily inventory_daily_vendor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.inventory_daily
    ADD CONSTRAINT inventory_daily_vendor_id_fkey FOREIGN KEY (vendor_id) REFERENCES public.vendor_master(vendor_id);


--
-- Name: inventory_master inventory_master_vendor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: meghanarendrasimha
--

ALTER TABLE ONLY public.inventory_master
    ADD CONSTRAINT inventory_master_vendor_id_fkey FOREIGN KEY (vendor_id) REFERENCES public.vendor_master(vendor_id);


--
-- PostgreSQL database dump complete
--

\unrestrict EPsTYz57IKv4gNtaicsI8O1c8r9GLHpSV2Qtvvg65MR6db9sL31C5ekcuGZzEeg

