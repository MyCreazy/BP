using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BPNerveNetWork
{
    /// <summary>
    /// MNerveNetWork类 
    /// </summary>
    public class MNerveNetWork
    {
        /// <summary>
        /// 输入数据
        /// </summary>
        private double[,] inputData = null;

        /// <summary>
        /// 教师数据
        /// </summary>
        private double[,] teacherData = null;

        /// <summary>
        /// 训练次数
        /// </summary>
        private int trainNumber = 0;

        /// <summary>
        /// 动量因子
        /// </summary>
        private double momentumFactor = 0;

        /// <summary>
        /// 精确度控制
        /// </summary>
        private double precisionControl = 0.0;

        /// <summary>
        /// 学习率
        /// </summary>
        private double studyRate = 0.0;

        /// <summary>
        /// 输出节点数目
        /// </summary>
        private int outputDataNumber = 0;

        /// <summary>
        /// 数据来源
        /// </summary>
        private string dataSource = string.Empty;

        /// <summary>
        /// 调整无效时调整学习率
        /// </summary>
        private double beta = 0.9;

        /// <summary>
        /// 调整有效时增大学习率
        /// </summary>
        private double theta = 1.1;

        /// <summary>
        /// 允许误差
        /// </summary>
        private double allowError = 0.03;

        /// <summary>
        /// 前一次的系统误差
        /// </summary>
        private double preSystemError = 30000;

        /// <summary>
        /// 前一次的系统误差
        /// </summary>
        public double PreSystemError
        {
            get { return this.preSystemError; }
            set { this.preSystemError = value; }
        }

        /// <summary>
        /// 允许误差
        /// </summary>
        public double AllowError
        {
            get { return this.allowError; }
            set { this.allowError = value; }
        }

        /// <summary>
        /// 调整有效时增大学习率
        /// </summary>
        public double Theta
        {
            get { return this.theta; }
            set { this.theta = value; }
        }  

        /// <summary>
        /// 调整无效时调整学习率
        /// </summary>
        public double Beta
        {
            get { return this.beta; }
            set { this.beta = value; }
        }

        /// <summary>
        /// 数据来源
        /// </summary>
        public string DataSource
        {
            get { return this.dataSource; }
            set { this.dataSource = value; }
        }

        /// <summary>
        /// 输出节点数目
        /// </summary>
        public int OutputDataNumber
        {
            get { return this.outputDataNumber; }
            set { this.outputDataNumber = value; }
        }

        /// <summary>
        /// 学习率
        /// </summary>
        public double StudyRate
        {
            get { return this.studyRate; }
            set { this.studyRate = value; }
        }

        /// <summary>
        /// 精确度控制
        /// </summary>
        public double PrecisionControl
        {
            get { return this.precisionControl; }
            set { this.precisionControl = value; }
        }

        /// <summary>
        /// 动量因子
        /// </summary>
        public double MomentumFactor
        {
            get { return this.momentumFactor; }
            set { this.momentumFactor = value; }
        }

        /// <summary>
        /// 训练次数
        /// </summary>
        public int TrainNumber
        {
            get { return this.trainNumber; }
            set { this.trainNumber = value; }
        }

        /// <summary>
        /// 教师数据
        /// </summary>
        public double[,] TeacherData
        {
            get { return this.teacherData; }
            set { this.teacherData = value; }
        }

        /// <summary>
        /// 输入数据
        /// </summary>
        public double[,] InputData
        {
            get { return this.inputData; }
            set { this.inputData = value; }
        }
    }
}
