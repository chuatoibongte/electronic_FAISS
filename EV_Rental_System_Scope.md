# SCOPE DEFINITION - Hệ Thống Thuê Xe Điện (EV Rental System)

## 📋 TỔNG QUAN SCOPE

### Mục tiêu dự án
Phát triển một hệ thống quản lý thuê xe điện tích hợp, hỗ trợ quy trình từ đăng ký, đặt xe, giao/nhận xe đến thanh toán và quản lý cho 3 nhóm người dùng chính.

---

## 🎯 FUNCTIONAL SCOPE (Phạm vi Chức năng)

### 1. **EV RENTER MODULE** (Người thuê xe)
**✅ TRONG SCOPE:**
- **Quản lý tài khoản:**
  - Đăng ký tài khoản mới
  - Upload và xác thực giấy phép lái xe, CMND/CCCD
  - Quản lý hồ sơ cá nhân
  
- **Đặt xe:**
  - Tìm kiếm điểm thuê trên bản đồ tích hợp
  - Xem danh sách xe available (loại xe, pin, giá)
  - Đặt xe trước hoặc walk-in
  - Hệ thống notification đặt xe
  
- **Quy trình thuê xe:**
  - Check-in qua app/tại quầy
  - Ký hợp đồng điện tử
  - Xác nhận bàn giao xe (chụp ảnh, ghi chú)
  - Trả xe và xác nhận tình trạng
  
- **Thanh toán & Chi phí:**
  - Thanh toán phí thuê
  - Quản lý đặt cọc/hoàn cọc
  - Thanh toán phí phát sinh
  
- **Lịch sử & Báo cáo cá nhân:**
  - Lịch sử thuê xe
  - Thống kê chi phí, quãng đường
  - Phân tích pattern sử dụng (giờ cao điểm/thấp điểm)

### 2. **STATION STAFF MODULE** (Nhân viên điểm thuê)
**✅ TRONG SCOPE:**
- **Quản lý giao/nhận xe:**
  - Dashboard xe available/booked/rented
  - Quy trình bàn giao xe (kiểm tra, chụp ảnh)
  - Ký xác nhận điện tử
  - Cập nhật trạng thái xe real-time
  
- **Xác thực khách hàng:**
  - Kiểm tra và đối chiếu giấy tờ
  - Xác thực danh tính qua hệ thống
  - Ghi nhận kết quả xác thực
  
- **Quản lý thanh toán tại điểm:**
  - Xử lý thanh toán phí thuê
  - Quản lý đặt cọc/hoàn cọc
  - In hóa đơn, biên lai
  
- **Quản lý xe tại station:**
  - Cập nhật trạng thái pin
  - Báo cáo tình trạng kỹ thuật
  - Ghi nhận sự cố lên hệ thống

### 3. **ADMIN MODULE** (Quản trị viên)
**✅ TRONG SCOPE:**
- **Quản lý đội xe & điểm thuê:**
  - Dashboard tổng quan số lượng xe
  - Theo dõi lịch sử giao/nhận
  - Điều phối xe giữa các điểm
  - Quản lý capacity và availability
  
- **Quản lý khách hàng:**
  - Database hồ sơ khách hàng
  - Lịch sử thuê và hành vi
  - Xử lý khiếu nại
  - Blacklist khách hàng có rủi ro
  
- **Quản lý nhân viên:**
  - Danh sách nhân viên các điểm
  - Phân công và scheduling
  - Theo dõi KPI (số lượt giao/nhận, satisfaction)
  - Đánh giá hiệu suất
  
- **Báo cáo & Analytics:**
  - Doanh thu theo điểm/thời gian
  - Tỷ lệ sử dụng xe
  - Phân tích giờ cao điểm/thấp điểm
  - Báo cáo tài chính
  - Predictive analytics cho demand

---

## 💻 TECHNICAL SCOPE (Phạm vi Kỹ thuật)

### **✅ TRONG SCOPE:**
- **Platform:**
  - Web Application (Responsive design)
  - Mobile App (iOS/Android) cho EV Renter
  - Admin Dashboard (Web-based)
  
- **Core Technologies:**
  - Backend API development
  - Database design và implementation
  - Authentication & Authorization system
  - File upload và storage system
  - Real-time notification system
  
- **Integration Requirements:**
  - Maps integration (Google Maps/OpenStreetMap)
  - Payment gateway integration
  - SMS/Email notification service
  - Digital signature capability
  - Photo capture và storage
  
- **Data Management:**
  - User data và document management
  - Vehicle tracking và status management
  - Transaction và payment records
  - Reporting và analytics engine

### **❌ NGOÀI SCOPE:**
- IoT integration với xe (GPS tracking, remote control)
- Advanced AI/ML features
- Multi-language support
- Third-party vehicle management systems
- Advanced CRM features

---

## 🏢 BUSINESS SCOPE (Phạm vi Kinh doanh)

### **✅ TRONG SCOPE:**
- **Business Model:**
  - B2C rental service
  - Station-based rental model
  - Hourly/daily pricing structure
  - Deposit và penalty system
  
- **Target Market:**
  - Individual consumers cần thuê xe điện
  - Urban areas với network các điểm thuê
  - Short-term rental (giờ/ngày)
  
- **Business Operations:**
  - Inventory management
  - Customer service workflow
  - Financial transaction processing
  - Operational reporting

### **❌ NGOÀI SCOPE:**
- B2B corporate contracts
- Long-term leasing (>30 days)
- Vehicle maintenance management
- Insurance claim processing
- Fleet expansion planning

---

## 🔒 DATA SCOPE (Phạm vi Dữ liệu)

### **✅ TRONG SCOPE:**
- **User Data:**
  - Personal information
  - Document verification data
  - Rental history và preferences
  
- **Vehicle Data:**
  - Basic vehicle information
  - Battery status và mileage
  - Availability và location
  
- **Transaction Data:**
  - Rental records
  - Payment transactions
  - Incident reports
  
- **Operational Data:**
  - Staff performance metrics
  - Station utilization
  - System usage analytics

### **❌ NGOÀI SCOPE:**
- Real-time vehicle telemetry
- Advanced behavioral analytics
- Third-party data integration
- Historical market data

---

## ⚖️ COMPLIANCE & SECURITY SCOPE

### **✅ TRONG SCOPE:**
- **Data Protection:**
  - Basic data encryption
  - User authentication
  - Role-based access control
  
- **Business Compliance:**
  - Basic audit trails
  - Transaction logging
  - Document retention policies

### **❌ NGOÀI SCOPE:**
- GDPR compliance (nếu không phục vụ EU)
- Advanced security certifications
- Blockchain integration
- Advanced fraud detection

---

## 📊 PERFORMANCE & SCALABILITY SCOPE

### **✅ TRONG SCOPE:**
- Support 1,000+ concurrent users
- Response time < 3 seconds cho core functions
- 99.5% uptime requirement
- Basic load balancing

### **❌ NGOÀI SCOPE:**
- Global scale deployment
- Advanced CDN integration
- Microservices architecture
- Advanced caching strategies

---

## 🚧 PROJECT CONSTRAINTS (Ràng buộc Dự án)

### **Time Constraints:**
- Development timeline: 6-12 months
- MVP delivery trong 4-6 months

### **Budget Constraints:**
- Focus on essential features first
- Scalable architecture cho future expansion

### **Resource Constraints:**
- Small to medium development team
- Standard development tools và frameworks

---

## 🎯 SUCCESS CRITERIA (Tiêu chí Thành công)

### **Functional Success:**
- ✅ Tất cả 3 user roles có thể thực hiện đầy đủ workflow
- ✅ End-to-end rental process hoạt động smooth
- ✅ Real-time data synchronization giữa các modules

### **Technical Success:**
- ✅ System stability và performance đạt yêu cầu
- ✅ Data integrity và security được đảm bảo
- ✅ User experience intuitive và responsive

### **Business Success:**
- ✅ Reduced manual processing time
- ✅ Improved customer satisfaction
- ✅ Better operational visibility và control

---

## 📋 DELIVERABLES (Sản phẩm Bàn giao)

### **Phase 1 - MVP:**
1. Core database design
2. Basic user authentication
3. Essential rental workflow
4. Admin dashboard basics

### **Phase 2 - Full Features:**
1. Complete mobile app
2. Advanced reporting
3. Integration với external services
4. Performance optimization

### **Phase 3 - Enhancement:**
1. Advanced analytics
2. Notification system
3. Document management
4. System monitoring

---

## ⚠️ ASSUMPTIONS & DEPENDENCIES

### **Assumptions:**
- Stable internet connectivity tại các stations
- Staff được training về hệ thống mới
- Customers có basic smartphone/internet skills

### **Dependencies:**
- Third-party payment gateway availability
- Maps API service reliability
- Cloud infrastructure stability
- Mobile app store approval process