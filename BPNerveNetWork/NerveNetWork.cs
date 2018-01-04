using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Better517Na.VerificationCodeParam.Bussiness;
using Better517Na.VerificationCodeParam.Model;

namespace BPNerveNetWork
{
    /// <summary>
    /// 神经网络
    /// </summary>
    public class NerveNetWork
    {
        #region 声明参数
        /// <summary>
        /// 输入节点数
        /// </summary>
        private int inputNodeNumber = 0;

        /// <summary>
        /// 隐藏层节点数
        /// </summary>
        private int hideNodeNumber = 0;

        /// <summary>
        /// 输出层节点数
        /// </summary>
        private int outPutNodeNumber = 0;

        /// <summary>
        /// 样本总数
        /// </summary>
        private int allSampleNumber = 0;

        /// <summary>
        /// 产生随机数对象
        /// </summary>
        private Random random = new Random();

        /// <summary>
        /// 隐藏层节点输出数据
        /// </summary>
        private double[,] hideNodeOutputData;

        /// <summary>
        /// 输出层节点输出数据
        /// </summary>
        private double[,] ouputNodeOutputData;

        /// <summary>
        /// 输出层权重修改量
        /// </summary>
        private double[] modifyOutWeight = null;

        /// <summary>
        /// 隐藏层权重修改量
        /// </summary>
        private double[] modifyHideWeight = null;

        /// <summary>
        /// 输入层到隐藏层的权值矩阵
        /// </summary>
        private double[,] hideWeightMatrix;

        /// <summary>
        /// 隐藏层到输出层的权值矩阵
        /// </summary>
        private double[,] outPutWeightMatrix;

        /// <summary>
        /// 上一次隐藏层权值调整量
        /// </summary>
        private double[,] lastHideWeight = null;

        /// <summary>
        /// 上一次输出层权值调整量
        /// </summary>
        private double[,] lastOutputWeight = null;

        /// <summary>
        /// 传进来的数据
        /// </summary>
        private MNerveNetWork data = null;

        /// <summary>
        /// 错误系数
        /// </summary>
        private double error = 0.0;

        /// <summary>
        /// 上一次的误差，方便进行学习率的调整
        /// </summary>
        private double lastError = 0.0;

        #region 新加
        /// <summary>
        /// 隐藏层阀值矩阵
        /// </summary>
        private double[] hideThreshold = null;

        /// <summary>
        /// 输出层阀值矩阵
        /// </summary>
        private double[] outputThreshold = null;

        /// <summary>
        /// 隐层误差
        /// </summary>
        private double[] hideError = null;

        /// <summary>
        /// 输出层误差
        /// </summary>
        private double[] ouputError = null;
        #endregion

        #endregion
        /// <summary>
        /// 构造函数初始化数据
        /// </summary>
        /// <param name="data">数据</param>
        public NerveNetWork(MNerveNetWork data)
        {
            this.random = new Random();
            this.data = data;

            // 错误系数
            this.error = 0;

            // 取得输入节点数目
            this.inputNodeNumber = this.data.InputData.GetLength(1);

            // 取得输出节点数目
            this.outPutNodeNumber = this.data.OutputDataNumber;

            // 得到隐藏节点数目
            this.hideNodeNumber = this.ComputeHideNumber(this.inputNodeNumber, this.outPutNodeNumber);

            // 样本总数
            this.allSampleNumber = this.data.InputData.GetLength(0);

            // 隐藏节点输出数据
            this.hideNodeOutputData = new double[this.allSampleNumber, this.hideNodeNumber];

            this.hideThreshold = new double[this.hideNodeNumber];
            this.hideError = new double[this.hideNodeNumber];
            for (int i = 0; i < this.hideNodeNumber; i++)
            {
                this.hideThreshold[i] = 0.0;
                this.hideError[i] = 0.0;
            }

            this.outputThreshold = new double[this.outPutNodeNumber];
            this.ouputError = new double[this.outPutNodeNumber];
            for (int k = 0; k < this.outPutNodeNumber; k++)
            {
                this.outputThreshold[k] = 0.0;
                this.ouputError[k] = 0.0;
            }

            for (int i = 0; i < this.allSampleNumber; i++)
            {
                for (int j = 0; j < this.hideNodeNumber; j++)
                {
                    this.hideNodeOutputData[i, j] = -1;
                }
            }

            // 输出节点输出数据
            this.ouputNodeOutputData = new double[this.allSampleNumber, this.outPutNodeNumber];

            for (int i = 0; i < this.allSampleNumber; i++)
            {
                for (int j = 0; j < this.outPutNodeNumber; j++)
                {
                    this.ouputNodeOutputData[i, j] = 0;
                }
            }

            // 上一次隐藏层权值调整量
            this.lastHideWeight = new double[this.inputNodeNumber, this.hideNodeNumber];
            for (int i = 0; i < this.inputNodeNumber; i++)
            {
                for (int j = 0; j < this.hideNodeNumber; j++)
                {
                    this.lastHideWeight[i, j] = 0;
                }
            }

            // 上一次输出层权值调整量
            this.lastOutputWeight = new double[this.hideNodeNumber, this.outPutNodeNumber];
            for (int i = 0; i < this.hideNodeNumber; i++)
            {
                for (int j = 0; j < this.outPutNodeNumber; j++)
                {
                    this.lastOutputWeight[i, j] = 0;
                }
            }

            this.hideWeightMatrix = new double[this.inputNodeNumber, this.hideNodeNumber];
            this.outPutWeightMatrix = new double[this.hideNodeNumber, this.outPutNodeNumber];

            // 初始化权值矩阵
            this.InitWeightMatrix(this.inputNodeNumber, this.hideNodeNumber, this.outPutNodeNumber);

            // 输出层权重修改量
            this.modifyOutWeight = new double[this.outPutNodeNumber];

            // 隐藏层权重修改量
            this.modifyHideWeight = new double[this.hideNodeNumber];
        }


        /// <summary>
        /// 无参构造函数
        /// </summary>
        public NerveNetWork()
        { }

        /// <summary>
        /// 训练神经网络
        /// </summary>
        public void TrainNetWork()
        {
            for (int i = 0; i < this.data.TrainNumber; i++)
            {
                this.error = 0.0;

                // 循环处理每个样本
                for (int j = 0; j < this.allSampleNumber; j++)
                {
                    // 计算隐藏层的输出
                    this.HideLayerOutput(this.hideNodeNumber, this.inputNodeNumber, j, this.hideThreshold);

                    // 计算输出层输出
                    this.OutputLayerOutput(this.hideNodeNumber, this.outPutNodeNumber, j, this.outputThreshold);

                    // 计算输出层误差和均方差
                    this.CountOutputLayerError(j);

                    // 计算隐藏层误差
                    this.CountHideLayerError(j);

                    // 更新输出层阀值矩阵
                    this.UpdateOutputThreshold();

                    // 更新隐藏层阀值矩阵
                    this.UpdateHideThreshold();
                }

                this.error = Math.Sqrt(this.error);
                if (this.error < this.data.AllowError)
                {
                    break;
                }

                if (this.lastError != 0 && this.error > this.lastError)
                {
                    this.data.StudyRate *= this.data.Beta;
                }
                else if (this.lastError != 0 && this.error < this.lastError)
                {
                    this.data.StudyRate *= this.data.Theta;
                }

                this.lastError = this.error;
            }

            // 保存取值矩阵
            this.SaveWeightMatrix();
        }

        /// <summary>
        /// 计算输出层误差
        /// </summary>
        /// <param name="sample"></param>
        private void CountOutputLayerError(int sample)
        {
            for (int i = 0; i < this.outPutNodeNumber; i++)
            {
                this.ouputError[i] = (this.data.TeacherData[sample, i] - this.ouputNodeOutputData[sample, i]) * this.ouputNodeOutputData[sample, i] * (1.0 - this.ouputNodeOutputData[sample, i]);
                this.error += (this.data.TeacherData[sample, i] - this.ouputNodeOutputData[sample, i]) * (this.data.TeacherData[sample, i] - this.ouputNodeOutputData[sample, i]);

                // 更新输出层权值矩阵
                for (int j = 0; j < this.hideNodeNumber; j++)
                {
                    this.outPutWeightMatrix[j, i] += this.data.StudyRate * this.ouputError[i] * this.hideNodeOutputData[sample, j] + this.data.MomentumFactor * this.lastOutputWeight[j, i];
                    this.lastOutputWeight[j, i] = this.data.StudyRate * this.ouputError[i] * this.hideNodeOutputData[sample, j];
                }
            }
        }

        /// <summary>
        /// 计算隐层误差
        /// </summary>
        /// <param name="sample"></param>
        private void CountHideLayerError(int sample)
        {
            //计算隐层误差
            for (int i = 0; i < this.hideNodeNumber; i++)
            {
                this.hideError[i] = 0.0;
                for (int j = 0; j < this.outPutNodeNumber; j++)
                {
                    this.hideError[i] += this.ouputError[j] * this.outPutWeightMatrix[i, j];
                }

                this.hideError[i] = this.hideError[i] * this.hideNodeOutputData[sample, i] * (1 - this.hideNodeOutputData[sample, i]);

                //更新w
                for (int k = 0; k < this.inputNodeNumber; k++)
                {
                    this.hideWeightMatrix[k, i] += this.data.StudyRate * this.hideError[i] * this.data.InputData[sample, k] + this.data.MomentumFactor * this.lastHideWeight[k, i];
                    this.lastHideWeight[k, i] = this.data.StudyRate * this.hideError[i] * this.data.InputData[sample, k];
                }
            }
        }

        /// <summary>
        /// 更新输出阀值矩阵
        /// </summary>
        private void UpdateOutputThreshold()
        {
            for (int i = 0; i < this.outPutNodeNumber; i++)
            {
                this.outputThreshold[i] += this.data.StudyRate * this.ouputError[i];
            }
        }

        /// <summary>
        /// 更新隐藏层阀值矩阵
        /// </summary>
        private void UpdateHideThreshold()
        {
            for (int i = 0; i < this.hideNodeNumber; i++)
            {
                this.hideThreshold[i] += this.data.StudyRate * this.hideError[i];
            }
        }

        /// <summary>
        /// 批量调整矩阵
        /// </summary>
        private void AdjustWeight()
        {
            // 隐藏层到输出层的权值矩阵调整大小
            double[,] adjustOutputWeight = new double[this.hideNodeNumber, this.outPutNodeNumber];

            // 输入层到隐藏层的权值矩阵调整大小
            double[,] adjustHideWeight = new double[this.inputNodeNumber, this.hideNodeNumber];

            // 调整隐藏层到输出层的权值矩阵
            for (int i = 0; i < this.hideNodeNumber; i++)
            {
                for (int j = 0; j < this.outPutNodeNumber; j++)
                {
                    // 遍历每个样本
                    for (int k = 0; k < this.allSampleNumber; k++)
                    {
                        adjustOutputWeight[i, j] += (this.data.TeacherData[k, j] - this.ouputNodeOutputData[k, j]) * this.ouputNodeOutputData[k, j] * (1 - this.ouputNodeOutputData[k, j]) * this.hideNodeOutputData[k, j];
                    }

                    adjustOutputWeight[i, j] *= this.data.StudyRate;

                    // 加入动量
                    this.outPutWeightMatrix[i, j] += adjustOutputWeight[i, j] + this.data.MomentumFactor * this.lastOutputWeight[i, j];
                    this.lastOutputWeight[i, j] = adjustOutputWeight[i, j];
                }
            }

            // 调整输入层到隐藏层的权值矩阵
            for (int i = 0; i < this.inputNodeNumber; i++)
            {
                for (int j = 0; j < this.hideNodeNumber; j++)
                {
                    for (int k = 0; k < this.allSampleNumber; k++)
                    {
                        double temp = 0.0;
                        for (int m = 0; m < this.outPutNodeNumber; m++)
                        {
                            temp += (this.data.TeacherData[k, m] - this.ouputNodeOutputData[k, m]) * this.ouputNodeOutputData[k, m] * (1 - this.ouputNodeOutputData[k, m]) * this.hideWeightMatrix[j, m];
                        }

                        adjustHideWeight[i, j] += temp * this.hideNodeOutputData[k, j] * (1 - this.hideNodeOutputData[k, j]) * this.data.InputData[k, i];
                    }

                    adjustHideWeight[i, j] *= this.data.StudyRate;
                    this.hideWeightMatrix[i, j] += adjustHideWeight[i, j] + this.data.StudyRate * this.lastHideWeight[i, j];
                    this.lastHideWeight[i, j] = adjustHideWeight[i, j];
                }
            }
        }

        /// <summary>
        /// 保存权值矩阵
        /// </summary>
        private void SaveWeightMatrix()
        {
            // 保存训练好的数据
            List<MWeightMatrix> saveWeightMatrix = new List<MWeightMatrix>();
            for (int i = 0; i < this.hideWeightMatrix.GetLength(0); i++)
            {
                string matrixLineData = string.Empty;
                MWeightMatrix heiddenWeightLine = new MWeightMatrix();
                for (int j = 0; j < this.hideWeightMatrix.GetLength(1); j++)
                {

                    matrixLineData += this.hideWeightMatrix[i, j].ToString();
                    if (j != this.hideWeightMatrix.GetLength(1) - 1)
                    {
                        matrixLineData += ",";
                    }
                }

                heiddenWeightLine.MatrixLineData = matrixLineData;
                heiddenWeightLine.DataSource = this.data.DataSource;
                heiddenWeightLine.MatrixLocation = i;
                heiddenWeightLine.MatrixType = 0;

                saveWeightMatrix.Add(heiddenWeightLine);
            }


            for (int i = 0; i < this.outPutWeightMatrix.GetLength(0); i++)
            {
                string matrixLineData = string.Empty;
                MWeightMatrix outputWeightLine = new MWeightMatrix();
                for (int j = 0; j < this.outPutWeightMatrix.GetLength(1); j++)
                {

                    matrixLineData += this.outPutWeightMatrix[i, j].ToString();
                    if (j != this.outPutWeightMatrix.GetLength(1) - 1)
                    {
                        matrixLineData += ",";
                    }
                }

                outputWeightLine.MatrixLineData = matrixLineData;
                outputWeightLine.DataSource = this.data.DataSource;
                outputWeightLine.MatrixLocation = i;
                outputWeightLine.MatrixType = 1;

                saveWeightMatrix.Add(outputWeightLine);
            }

            MWeightMatrix hideThresholdeWeight = new MWeightMatrix();
            string matrixLineDataOne = string.Empty;
            for (int i = 0; i < this.hideThreshold.GetLength(0); i++)
            {
                matrixLineDataOne += this.hideThreshold[i].ToString();
                if (i != this.hideThreshold.GetLength(0) - 1)
                {
                    matrixLineDataOne += ",";
                }
            }

            hideThresholdeWeight.MatrixLineData = matrixLineDataOne;
            hideThresholdeWeight.DataSource = this.data.DataSource;
            hideThresholdeWeight.MatrixLocation = 0;
            hideThresholdeWeight.MatrixType = 2;
            saveWeightMatrix.Add(hideThresholdeWeight);



            MWeightMatrix outPutThresholdeWeight = new MWeightMatrix();
            string matrixLineDataTwo = string.Empty;
            for (int i = 0; i < this.outputThreshold.GetLength(0); i++)
            {
                matrixLineDataTwo += this.outputThreshold[i].ToString();
                if (i != this.outputThreshold.GetLength(0) - 1)
                {
                    matrixLineDataTwo += ",";
                }
            }

            outPutThresholdeWeight.MatrixLineData = matrixLineDataTwo;
            outPutThresholdeWeight.DataSource = this.data.DataSource;
            outPutThresholdeWeight.MatrixLocation = 0;
            outPutThresholdeWeight.MatrixType = 3;
            saveWeightMatrix.Add(outPutThresholdeWeight);



            // 保存
            VerficationCodeParamOperate operate = new VerficationCodeParamOperate();
            operate.UpdateWeightMatrix(saveWeightMatrix);
        }

        /// <summary>
        /// 神经网络仿真
        /// </summary>  
        /// <param name="samPle">测试数据</param>
        /// <param name="hideWeightMatrix">隐层权值矩阵</param>
        /// <param name="outPutWeightMatrix">输出层权值矩阵</param>
        /// <param name="inputNodeNumber">输入节点</param>
        /// <param name="hideNodeNumber">隐层节点</param>
        /// <param name="outPutNodeNumber">输出层节点</param>
        /// <param name="location">位置</param>
        /// <returns>输出值</returns>
        public double[] NetWorkSim(double[,] samPle, double[,] hideWeightMatrix, double[,] outPutWeightMatrix, double[] hideThreshold, double[] outPutThreshold, int inputNodeNumber, int hideNodeNumber, int outPutNodeNumber, int location)
        {
            double[] O1 = new double[hideNodeNumber];
            double[] O2 = new double[outPutNodeNumber];

            //计算隐藏层输出 
            for (int i = 0; i < hideNodeNumber; ++i)
            {
                double temp = 0;
                for (int j = 0; j < inputNodeNumber; ++j)
                {
                    temp += samPle[location, j] * hideWeightMatrix[j, i];
                }

                O1[i] = this.IncentiveFunction(temp, hideThreshold[i]);
            }

            //计算输出层输出 
            for (int i = 0; i < outPutNodeNumber; ++i)
            {
                double temp = 0;
                for (int j = 0; j < hideNodeNumber; ++j)
                {
                    temp += O1[j] * outPutWeightMatrix[j, i];
                }

                O2[i] = this.IncentiveFunction(temp, outPutThreshold[i]);
            }

            return O2;
        }

        /// <summary>
        /// 计算隐藏节点数目
        /// </summary>
        /// <param name="inputNodeNumber">输入节点数目</param>
        /// <param name="ouputNodeNumber">输出节点数目</param>
        /// <returns>返回隐藏节点数目</returns>
        private int ComputeHideNumber(int inputNodeNumber, int ouputNodeNumber)
        {
            int node = 0;
            node = Convert.ToInt32(Math.Sqrt(Convert.ToDouble(inputNodeNumber + outPutNodeNumber))) + 5;
            return node;
        }

        /// <summary>
        /// sigmoid 函数,神经网络激励函数 
        /// </summary>
        /// <param name="netValue">输出值</param>
        /// <returns>返回输出值</returns>
        private double IncentiveFunction(double netValue, double threshold)
        {
            return 1.0 / (1 + Math.Exp(-1 * netValue - threshold));
        }

        ///// <summary>
        ///// 修改隐藏层权值矩阵
        ///// </summary>
        ///// <param name="sampleLocation">样本位置</param>
        //private void ModifyHideWeightMatrix(int sampleLocation)
        //{
        //    for (int i = 0; i < this.inputNodeNumber; i++)
        //    {
        //        for (int j = 0; j < this.hideNodeNumber; j++)
        //        {
        //            this.hideWeightMatrix[i, j] = this.hideWeightMatrix[i, j] + this.data.StudyRate * this.data.TeacherData[sampleLocation] * this.modifyHideWeight[j];
        //        }
        //    }
        //}

        ///// <summary>
        ///// 修改输出层权值矩阵
        ///// </summary>
        //private void ModifyOutWeightMatrix()
        //{
        //    for (int i = 0; i < this.hideNodeNumber; i++)
        //    {
        //        for (int j = 0; j < this.outPutNodeNumber; j++)
        //        {
        //            this.outPutWeightMatrix[i, j] = this.data.MomentumFactor * this.outPutWeightMatrix[i, j] + this.data.StudyRate * this.hideNodeOutputData[i] * this.modifyOutWeight[j];
        //        }
        //    }
        //}

        ///// <summary>
        ///// 计算隐藏层的修改量
        ///// </summary>
        //private void ComputeHideWeightModify()
        //{
        //    for (int i = 0; i < this.hideNodeNumber; i++)
        //    {
        //        double temp = 0.0;
        //        for (int j = 0; j < this.outPutNodeNumber; j++)
        //        {
        //            temp += this.modifyOutWeight[j] * this.outPutWeightMatrix[i, j];
        //        }

        //        double tempOne = temp * this.hideNodeOutputData[i] * (1 - this.hideNodeOutputData[i]);
        //        this.modifyHideWeight[i] = this.data.MomentumFactor * this.modifyHideWeight[i] + (1 - this.data.MomentumFactor) * tempOne;
        //    }
        //}

        /// <summary>
        /// 计算输出误差
        /// </summary>
        /// <param name="sampleLocation">样本位置</param>
        private void ComputeOutputError(int sampleLocation)
        {
            for (int i = 0; i < this.outPutNodeNumber; i++)
            {
                this.error += (this.data.TeacherData[sampleLocation, i] - this.ouputNodeOutputData[sampleLocation, i]) * (this.data.TeacherData[sampleLocation, i] - this.ouputNodeOutputData[sampleLocation, i]);
            }
        }

        ///// <summary>
        ///// 计算输出层权重修改量
        ///// </summary>
        ///// <param name="sampleLocation">样本位置</param>
        //private void ComputeOutWeightModify(int sampleLocation)
        //{
        //    for (int i = 0; i < this.outPutNodeNumber; i++)
        //    {
        //        // 使用 momentumFactor 动量因子，避免陷入局部最优，momentumFactor设置为0.9 
        //        double temp = this.ouputNodeOutputData[i] * (1 - this.ouputNodeOutputData[i]) * (this.data.TeacherData[sampleLocation] - this.ouputNodeOutputData[i]);
        //        this.modifyOutWeight[i] = this.data.MomentumFactor * this.modifyOutWeight[i] + (1 - this.data.MomentumFactor) * temp;
        //    }
        //}

        /// <summary>
        /// 计算隐藏节点输出
        /// </summary>
        /// <param name="hideNodeNumber">隐藏节点数目</param>
        /// <param name="inputNodeNumber">输入节点数目</param>
        /// <param name="sampleLocation">第几个样本</param>
        private void HideLayerOutput(int hideNodeNumber, int inputNodeNumber, int sampleLocation, double[] hideThresholde)
        {
            for (int i = 0; i < hideNodeNumber; i++)
            {
                double tempNumber = 0;
                for (int j = 0; j < inputNodeNumber; j++)
                {
                    tempNumber += this.data.InputData[sampleLocation, j] * this.hideWeightMatrix[j, i];
                }

                this.hideNodeOutputData[sampleLocation, i] = this.IncentiveFunction(tempNumber, hideThresholde[i]);
            }
        }

        /// <summary>
        /// 计算输出层节点输出
        /// </summary>
        /// <param name="hideNodeNumber">隐藏层节点数</param>
        /// <param name="outNodeNumber">输出层节点数</param>
        /// <param name="sampleLocation">第几个样本</param>
        private void OutputLayerOutput(int hideNodeNumber, int outNodeNumber, int sampleLocation, double[] outputThreshold)
        {
            for (int i = 0; i < outNodeNumber; i++)
            {
                double tempNumber = 0;
                for (int j = 0; j < hideNodeNumber; j++)
                {
                    tempNumber += this.hideNodeOutputData[sampleLocation, j] * this.outPutWeightMatrix[j, i];
                }

                this.ouputNodeOutputData[sampleLocation, i] = this.IncentiveFunction(tempNumber, outputThreshold[i]);
            }
        }

        /// <summary>
        /// 初始化权值矩阵
        /// </summary>
        /// <param name="inDimension">输入节点数目</param>
        /// <param name="hidedenDimension">隐藏层节点数</param>
        /// <param name="outDimension">输出层节点数</param>
        private void InitWeightMatrix(int inDimension, int hidedenDimension, int outDimension)
        {
            // 变成从数据库获取，如果没有获取到数据，则随机产生
            VerficationCodeParamOperate operate = new VerficationCodeParamOperate();
            List<MWeightMatrix> hiddenmatrix = operate.GetWeightMatrix("MU", 0);
            hiddenmatrix = hiddenmatrix.OrderBy(p => p.MatrixLocation).ToList();

            List<MWeightMatrix> outputmatrix = operate.GetWeightMatrix("MU", 1);
            outputmatrix = outputmatrix.OrderBy(p => p.MatrixLocation).ToList();

            // 初始化输入到隐藏层的权值矩阵
            for (int i = 0; i < inDimension; i++)
            {
                string[] dd = null;

                // 存的数据以逗号隔开
                if (hiddenmatrix != null && hiddenmatrix.Count > 0)
                {
                    dd = hiddenmatrix[i].MatrixLineData.Split(',');
                }

                for (int j = 0; j < hidedenDimension; j++)
                {
                    if (dd != null)
                    {
                        this.hideWeightMatrix[i, j] = Convert.ToDouble(dd[j]);
                    }
                    else
                    {
                        this.hideWeightMatrix[i, j] = (2.0 * this.random.NextDouble()) - 1.0;
                    }
                }
            }

            // 初始化隐藏层到输出层的权值矩阵
            for (int i = 0; i < hidedenDimension; i++)
            {
                string[] dd = null;

                // 存的数据以逗号隔开
                if (outputmatrix != null && outputmatrix.Count > 0)
                {
                    dd = outputmatrix[i].MatrixLineData.Split(',');
                }

                for (int j = 0; j < outDimension; j++)
                {
                    if (dd != null)
                    {
                        this.outPutWeightMatrix[i, j] = Convert.ToDouble(dd[j]);

                    }
                    else
                    {
                        this.outPutWeightMatrix[i, j] = (2.0 * this.random.NextDouble()) - 1.0;
                    }
                }
            }
        }
    }
}
