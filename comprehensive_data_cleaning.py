import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def comprehensive_clean_healthcare_data():
    """
    综合版医疗数据清洗
    流程：先删除所有缺失值行，再进行增强清洗和特征工程
    """
    print("开始综合版数据清洗 - 先删除缺失值，再增强清洗...")
    
    # 读取原始数据
    df = pd.read_csv('healthcare/train_data.csv')
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据总缺失值: {df.isnull().sum().sum()}")
    
    # 第一步：删除所有包含缺失值的行
    print("\n=== 第一步：删除所有包含缺失值的行 ===")
    df_cleaned = df.dropna()
    print(f"删除缺失值行后数据形状: {df_cleaned.shape}")
    print(f"删除的行数: {len(df) - len(df_cleaned)}")
    print(f"数据保留率: {(len(df_cleaned) / len(df) * 100):.2f}%")
    
    # 验证删除效果
    remaining_missing = df_cleaned.isnull().sum().sum()
    print(f"剩余缺失值数量: {remaining_missing}")
    
    if remaining_missing == 0:
        print("✓ 所有缺失值已通过删除行处理")
    else:
        print("⚠ 仍有缺失值存在")
    
    # 第二步：删除Stay列中值为"2025/11/20"的行
    print("\n=== 第二步：删除Stay列中值为'2025/11/20'的行 ===")
    stay_date_rows = df_cleaned[df_cleaned['Stay'] == '2025/11/20']
    print(f"找到Stay值为'2025/11/20'的行数: {len(stay_date_rows)}")
    
    if len(stay_date_rows) > 0:
        df_cleaned = df_cleaned[df_cleaned['Stay'] != '2025/11/20']
        print(f"删除Stay日期行后数据形状: {df_cleaned.shape}")
        print(f"删除的行数: {len(stay_date_rows)}")
        print(f"数据保留率: {(len(df_cleaned) / len(df) * 100):.2f}%")
    else:
        print("✓ 未发现Stay值为'2025/11/20'的行")
    
    # 第二步：数据类型转换和特征工程
    print("\n=== 第二步：数据类型转换和特征工程 ===")
    
    # Age列转换为数值型（取中间值）
    age_mapping = {
        '0-10': 5, '11-20': 15, '21-30': 25, '31-40': 35, '41-50': 45,
        '51-60': 55, '61-70': 65, '71-80': 75, '81-90': 85, '91-100': 95
    }
    df_cleaned['Age_numeric'] = df_cleaned['Age'].map(age_mapping)
    print("已将Age列转换为数值型 (Age_numeric)")
    
    # 创建年龄分组特征
    def age_group(age_str):
        if age_str in ['0-10', '11-20', '21-30']:
            return 'Young'
        elif age_str in ['31-40', '41-50', '51-60']:
            return 'Middle'
        else:
            return 'Senior'
    
    df_cleaned['Age_Group'] = df_cleaned['Age'].apply(age_group)
    print("新增年龄分组特征: Age_Group")
    
    # Stay列转换为数值型（取范围中位数）
    stay_mapping = {
        '0-10': 5.5, '11-20': 15.5, '21-30': 25.5, '31-40': 35.5, '41-50': 45.5,
        '51-60': 55.5, '61-70': 65.5, '71-80': 75.5, '81-90': 85.5, '91-100': 95.5,
        'More than 100 Days': 120  # 使用120作为更合理的估计值
    }
    df_cleaned['Stay_numeric'] = df_cleaned['Stay'].map(stay_mapping)
    print("已将Stay列转换为数值型 (Stay_numeric) - 使用中位数")
    
    # 新增：平均日访客特征
    df_cleaned['Daily_Visitors_Rate'] = df_cleaned['Visitors with Patient'] / df_cleaned['Stay_numeric']
    print("新增平均日访客特征: Daily_Visitors_Rate")
    
    # 新增：病人城市与医院城市关联特征
    print("\n=== 新增城市关联特征 ===")
    
    # 计算每个城市的病人总数
    city_patient_counts = df_cleaned.groupby('City_Code_Patient').size()
    
    # 计算每个城市中在本城医院就医的病人数
    # 对于城市a，计算在a城医院就医的a城病人数
    same_city_treatment_counts = {}
    for city_code in city_patient_counts.index:
        # 找到病人城市为a且医院城市也为a的记录
        same_city_count = len(df_cleaned[
            (df_cleaned['City_Code_Patient'] == city_code) & 
            (df_cleaned['City_Code_Hospital'] == city_code)
        ])
        same_city_treatment_counts[city_code] = same_city_count
    
    # 计算本城病人流失率
    city_loss_rate = {}
    for city_code in city_patient_counts.index:
        total_patients = city_patient_counts[city_code]
        same_city_treatment_count = same_city_treatment_counts[city_code]
        loss_rate = 1 - (same_city_treatment_count / total_patients) if total_patients > 0 else 0
        city_loss_rate[city_code] = loss_rate
        print(f"城市{city_code}: 总病人数={total_patients}, 本城就医数={same_city_treatment_count}, 流失率={loss_rate:.4f}")
    
    # 添加本城病人流失率特征
    df_cleaned['City_Patient_Loss_Rate'] = df_cleaned['City_Code_Patient'].map(city_loss_rate)
    print("新增本城病人流失率特征: City_Patient_Loss_Rate")
    
    # 添加是否在本城就医的标记
    df_cleaned['Same_City_Treatment'] = (df_cleaned['City_Code_Patient'] == df_cleaned['City_Code_Hospital']).astype(int)
    print("新增是否在本城就医标记: Same_City_Treatment")
    
    # 移除医院规模分组（保留原始房间数，避免信息丢失）
    print("已移除医院规模分组，保留原始房间数信息")
    
    # 病情严重程度编码
    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Extreme': 3}
    df_cleaned['Severity_encoded'] = df_cleaned['Severity of Illness'].map(severity_mapping)
    print("新增病情严重程度编码: Severity_encoded")
    
    # 第三步：数据标准化
    print("\n=== 第三步：数据标准化 ===")
    
    # 对数值型特征进行标准化
    numeric_cols = ['Admission_Deposit', 'Visitors with Patient', 'Age_numeric']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cleaned[numeric_cols])
    
    for i, col in enumerate(numeric_cols):
        df_cleaned[f'{col}_scaled'] = scaled_features[:, i]
        print(f"新增标准化特征: {col}_scaled")
    
    # 第四步：最终数据质量报告
    print("\n=== 第四步：最终数据质量报告 ===")
    print(f"最终数据形状: {df_cleaned.shape}")
    print(f"总缺失值数量: {df_cleaned.isnull().sum().sum()}")
    print(f"新增特征数量: {df_cleaned.shape[1] - 18}")
    
    # 显示新增的列
    original_cols = [
        'case_id', 'Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 
        'Hospital_region_code', 'Available Extra Rooms in Hospital', 'Department', 
        'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'patientid', 
        'City_Code_Patient', 'Type of Admission', 'Severity of Illness', 
        'Visitors with Patient', 'Age', 'Admission_Deposit', 'Stay'
    ]
    new_columns = set(df_cleaned.columns) - set(original_cols)
    print(f"新增列名: {list(new_columns)}")
    
    return df_cleaned

def save_comprehensive_cleaned_data(df_comprehensive_cleaned):
    """保存综合版清洗后的数据"""
    output_file = 'healthcare/train_data_comprehensive_cleaned.csv'
    df_comprehensive_cleaned.to_csv(output_file, index=False)
    print(f"\n综合版清洗后的数据已保存到: {output_file}")
    return output_file

if __name__ == "__main__":
    # 执行综合版数据清洗
    comprehensive_cleaned_df = comprehensive_clean_healthcare_data()
    
    # 保存综合版清洗后的数据
    output_path = save_comprehensive_cleaned_data(comprehensive_cleaned_df)
    
    print(f"\n综合版数据清洗完成！")
    print(f"原始数据行数: 318,438")
    print(f"清洗后数据行数: {len(comprehensive_cleaned_df):,}")
    print(f"删除的行数: {318438 - len(comprehensive_cleaned_df):,}")
    print(f"数据保留率: {(len(comprehensive_cleaned_df) / 318438 * 100):.2f}%")
    print(f"原始数据列数: 18")
    print(f"清洗后数据列数: {comprehensive_cleaned_df.shape[1]}")
    print(f"新增特征数量: {comprehensive_cleaned_df.shape[1] - 18}")
    print(f"数据完整性: 100.00%")
