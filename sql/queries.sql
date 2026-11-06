-- sql/queries.sql: 7 Analytical SQL Queries for Instamart Customer Intelligence

-- Q1: Completed orders with order_amount > 500, grouped by customer, having total > 8000
-- Reference: 30 customers
SELECT 
    customer_id, 
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Q2: Revenue by city and product category (join orders, customers, products)
-- Reference: 30 rows, sum INR 2,998,668.80, largest cell Bengaluru / Electronics at INR 233,909.45
SELECT 
    c.city, 
    p.category, 
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Products priced above the average catalogue price (scalar subquery)
-- Reference: 11 products
SELECT 
    product_id, 
    product_name, 
    category, 
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: Customers who have no completed order (LEFT JOIN)
-- Reference: 25 customers
SELECT 
    c.customer_id, 
    c.city, 
    c.signup_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Customers with total completed spend above average customer spend (Common Table Expression)
-- Reference: 550 customers, average INR 2,552.06
WITH customer_spend AS (
    SELECT 
        customer_id, 
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total_spend
    FROM customer_spend
)
SELECT 
    cs.customer_id, 
    ROUND(cs.total_spend, 2) AS total_spend
FROM customer_spend cs
CROSS JOIN avg_spend a
WHERE cs.total_spend > a.avg_total_spend
ORDER BY cs.total_spend DESC;

-- Q6: Ranking window functions for completed spend per customer partitioned by city
-- Reference: Top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60),
-- Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10), exactly 8 rows have RANK() != ROW_NUMBER()
WITH cust_city_spend AS (
    SELECT 
        c.customer_id, 
        c.city, 
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT 
    customer_id, 
    city, 
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM cust_city_spend
ORDER BY city, total_spend DESC;

-- Q7: Daily completed revenue with running total
-- Reference: 183 daily rows, last cumulative value INR 2,998,668.80
WITH daily_revenue AS (
    SELECT 
        SUBSTR(order_ts, 1, 10) AS day, 
        ROUND(SUM(order_amount), 2) AS daily_rev
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    day, 
    daily_rev,
    ROUND(SUM(daily_rev) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
