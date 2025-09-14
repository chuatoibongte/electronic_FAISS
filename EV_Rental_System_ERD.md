# Conceptual ERD - Hệ Thống Thuê Xe Điện (EV Rental System)

## Mô tả hệ thống
Hệ thống quản lý thuê xe điện với 3 nhóm người dùng chính:
- **EV Renter**: Người thuê xe
- **Station Staff**: Nhân viên tại điểm thuê  
- **Admin**: Quản trị viên hệ thống

## Conceptual ERD Diagram

```mermaid
erDiagram
    USER {
        string user_id PK
        string username
        string email
        string phone
        string full_name
        date date_of_birth
        string address
        string user_type
        string status
        datetime created_at
        datetime updated_at
    }
    
    DOCUMENT {
        string document_id PK
        string user_id FK
        string document_type
        string document_number
        string issued_by
        date issued_date
        date expiry_date
        string file_path
        string verification_status
        datetime created_at
    }
    
    STATION {
        string station_id PK
        string station_name
        string address
        decimal latitude
        decimal longitude
        string contact_phone
        string operating_hours
        integer capacity
        string status
        datetime created_at
    }
    
    VEHICLE {
        string vehicle_id PK
        string station_id FK
        string license_plate
        string brand
        string model
        string vehicle_type
        integer battery_capacity
        decimal current_battery_level
        decimal mileage
        string status
        decimal rental_price_per_hour
        datetime last_maintenance
        datetime created_at
    }
    
    STAFF_ASSIGNMENT {
        string assignment_id PK
        string user_id FK
        string station_id FK
        string role
        date start_date
        date end_date
        string status
    }
    
    RENTAL {
        string rental_id PK
        string renter_id FK
        string vehicle_id FK
        string station_id FK
        string staff_id FK
        datetime booking_time
        datetime start_time
        datetime end_time
        datetime actual_return_time
        string rental_status
        decimal total_amount
        decimal deposit_amount
        string contract_signed
        datetime created_at
    }
    
    VEHICLE_INSPECTION {
        string inspection_id PK
        string rental_id FK
        string vehicle_id FK
        string inspector_id FK
        string inspection_type
        datetime inspection_time
        decimal battery_level_before
        decimal battery_level_after
        decimal mileage_before
        decimal mileage_after
        text damage_notes
        text photos_path
        string condition_status
    }
    
    PAYMENT {
        string payment_id PK
        string rental_id FK
        string payer_id FK
        decimal amount
        string payment_type
        string payment_method
        string payment_status
        datetime payment_time
        string transaction_reference
        text notes
    }
    
    INCIDENT_REPORT {
        string incident_id PK
        string rental_id FK
        string vehicle_id FK
        string reporter_id FK
        string incident_type
        text description
        datetime incident_time
        datetime reported_time
        string severity
        decimal estimated_cost
        string status
        text resolution_notes
    }
    
    RENTAL_HISTORY {
        string history_id PK
        string user_id FK
        string rental_id FK
        decimal total_distance
        decimal total_cost
        integer total_duration_hours
        datetime trip_start
        datetime trip_end
        text feedback
        integer rating
    }

    %% Relationships
    USER ||--o{ DOCUMENT : "has"
    USER ||--o{ STAFF_ASSIGNMENT : "assigned_to"
    USER ||--o{ RENTAL : "rents_as_customer"
    USER ||--o{ RENTAL : "handles_as_staff"
    USER ||--o{ VEHICLE_INSPECTION : "inspects"
    USER ||--o{ PAYMENT : "makes"
    USER ||--o{ INCIDENT_REPORT : "reports"
    USER ||--o{ RENTAL_HISTORY : "has_history"
    
    STATION ||--o{ VEHICLE : "houses"
    STATION ||--o{ STAFF_ASSIGNMENT : "has_staff"
    STATION ||--o{ RENTAL : "location_for"
    
    VEHICLE ||--o{ RENTAL : "rented_in"
    VEHICLE ||--o{ VEHICLE_INSPECTION : "inspected"
    VEHICLE ||--o{ INCIDENT_REPORT : "involved_in"
    
    RENTAL ||--|| VEHICLE_INSPECTION : "has_pickup_inspection"
    RENTAL ||--|| VEHICLE_INSPECTION : "has_return_inspection"
    RENTAL ||--o{ PAYMENT : "involves"
    RENTAL ||--o{ INCIDENT_REPORT : "may_have"
    RENTAL ||--|| RENTAL_HISTORY : "generates"
```

## Mô tả các Entity chính

### 1. USER
- **Mục đích**: Lưu trữ thông tin tất cả người dùng trong hệ thống
- **Các loại**: EV Renter, Station Staff, Admin
- **Thuộc tính quan trọng**: user_type để phân biệt vai trò

### 2. DOCUMENT  
- **Mục đích**: Quản lý giấy tờ tùy thân, giấy phép lái xe
- **Liên kết**: Mỗi USER có thể có nhiều DOCUMENT
- **Xác thực**: verification_status để theo dõi trạng thái xác thực

### 3. STATION
- **Mục đích**: Quản lý các điểm thuê xe
- **Vị trí**: Có tọa độ GPS để hiển thị trên bản đồ
- **Capacity**: Số lượng xe tối đa có thể chứa

### 4. VEHICLE
- **Mục đích**: Quản lý đội xe điện
- **Trạng thái**: Available, Rented, Maintenance, Out_of_Service
- **Pin**: Theo dõi mức pin hiện tại và dung lượng

### 5. STAFF_ASSIGNMENT
- **Mục đích**: Quản lý việc phân công nhân viên tại các điểm
- **Linh hoạt**: Cho phép nhân viên làm việc tại nhiều điểm khác nhau

### 6. RENTAL
- **Mục đích**: Ghi nhận các giao dịch thuê xe
- **Quy trình**: Từ booking → start → end → return
- **Hợp đồng**: contract_signed để xác nhận ký hợp đồng điện tử

### 7. VEHICLE_INSPECTION
- **Mục đích**: Kiểm tra tình trạng xe khi giao và nhận
- **Loại**: Pickup inspection và Return inspection
- **Bằng chứng**: Lưu trữ ảnh chụp tình trạng xe

### 8. PAYMENT
- **Mục đích**: Quản lý các giao dịch thanh toán
- **Linh hoạt**: Hỗ trợ nhiều phương thức thanh toán
- **Loại**: Rental fee, Deposit, Refund, Penalty

### 9. INCIDENT_REPORT
- **Mục đích**: Ghi nhận sự cố, hư hỏng trong quá trình thuê
- **Theo dõi**: Từ báo cáo → xử lý → giải quyết
- **Chi phí**: Ước tính chi phí sửa chữa

### 10. RENTAL_HISTORY
- **Mục đích**: Lưu trữ lịch sử thuê xe của khách hàng
- **Phân tích**: Hỗ trợ báo cáo và phân tích hành vi khách hàng
- **Đánh giá**: Rating và feedback từ khách hàng

## Các mối quan hệ quan trọng

1. **USER - RENTAL**: Một người dùng có thể có nhiều lần thuê (1:N)
2. **VEHICLE - RENTAL**: Một xe có thể được thuê nhiều lần (1:N)  
3. **STATION - VEHICLE**: Một điểm có thể chứa nhiều xe (1:N)
4. **RENTAL - VEHICLE_INSPECTION**: Mỗi lần thuê có 2 lần kiểm tra (1:2)
5. **RENTAL - PAYMENT**: Một lần thuê có thể có nhiều khoản thanh toán (1:N)

## Lưu ý thiết kế

- **Flexibility**: Thiết kế linh hoạt để hỗ trợ mở rộng tương lai
- **Audit Trail**: Tất cả entity đều có timestamp để theo dõi
- **Status Management**: Sử dụng status fields để quản lý trạng thái
- **Data Integrity**: Sử dụng Foreign Keys để đảm bảo tính toàn vẹn dữ liệu