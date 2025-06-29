import os
import math
import requests
from typing import Optional
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.tools import StructuredTool
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from urllib.parse import quote

# ----------------------------
# 加载环境变量
# ----------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("❌ 警告：未检测到 DEEPSEEK_API_KEY，请检查 .env 文件或系统环境变量设置")
else:
    print("✅ 检测到 DEEPSEEK_API_KEY")

# ----------------------------
# 计算器工具
# ----------------------------

class CalculatorInput(BaseModel):
    operation: str = Field(description="Mathematical operation: add, subtract, multiply, divide, power, sqrt")
    a: float = Field(description="First number")
    b: Optional[float] = Field(default=None, description="Second number (not required for sqrt)")

def calculator(operation: str, a: float, b: Optional[float] = None) -> str:
    op = operation.lower()
    if op == "add":
        return f"{a} + {b} = {a + b}"
    elif op == "subtract":
        return f"{a} - {b} = {a - b}"
    elif op == "multiply":
        return f"{a} * {b} = {a * b}"
    elif op == "divide":
        if b == 0:
            return "Error: Division by zero"
        return f"{a} / {b} = {a / b}"
    elif op == "power":
        return f"{a}^{b} = {a ** b}"
    elif op == "sqrt":
        if a < 0:
            return "Error: Cannot take square root of negative number"
        return f"sqrt({a}) = {math.sqrt(a)}"
    else:
        return f"Unknown operation: {op}"

calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="Calculator",
    description="用于执行基本数学计算（加减乘除、平方、开方）",
    args_schema=CalculatorInput
)

# ----------------------------
# 天气工具
# ----------------------------

class WeatherInput(BaseModel):
    city: str = Field(description="要查询天气的城市名称")

def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "❌ 错误：未配置 OPENWEATHER_API_KEY 环境变量"
    
    encoded_city = quote(city)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("message", "未知错误")
            if error_msg == "city not found":
                # 回退尝试使用北京
                pinyin_url = f"https://api.openweathermap.org/data/2.5/weather?q=Beijing&appid={api_key}&units=metric"
                pinyin_response = requests.get(pinyin_url)
                if pinyin_response.status_code == 200:
                    return get_weather_data("Beijing", pinyin_response.json())
                return f"❌ 未找到城市: {city}。请尝试使用拼音名称(如: Beijing)"
            return f"❌ 天气查询失败: {error_msg} (状态码 {response.status_code})"
        
        return get_weather_data(city, data)

    except Exception as e:
        return f"❌ 请求异常: {str(e)}"

def get_weather_data(city: str, data: dict) -> str:
    try:
        weather_desc = data["weather"][0]["description"]
        main_data = data["main"]
        wind_data = data["wind"]

        return (
            f"📍 {city} 当前天气：\n"
            f"- 天气状况: {weather_desc}\n"
            f"- 温度: {main_data.get('temp', 'N/A')}°C\n"
            f"- 体感温度: {main_data.get('feels_like', 'N/A')}°C\n"
            f"- 湿度: {main_data.get('humidity', 'N/A')}%\n"
            f"- 风速: {wind_data.get('speed', 'N/A')} m/s"
        )
    except KeyError:
        return "❌ 解析天气数据时出错: API响应格式异常"

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="Weather",
    description="查询指定城市的当前天气信息",
    args_schema=WeatherInput
)

# ----------------------------
# 创建 Agent
# ----------------------------

def create_agent():
    """创建并返回配置好的Agent"""
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.1
    )

    tools = [calculator_tool, weather_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent

# ----------------------------
# CLI 主程序
# ----------------------------

if __name__ == "__main__":
    print("🤖 Multi-Tool Agent (Calculator + Weather)\n输入 'exit' 退出")

    try:
        agent = create_agent()
        print("✅ Agent 初始化成功！")
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        exit(1)

    while True:
        query = input("\n请输入你的问题: ")
        if query.strip().lower() == "exit":
            print("再见！")
            break

        try:
            result = agent.invoke({"input": query})
            print(f"\n🧠 回复结果:\n{result['output']}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
