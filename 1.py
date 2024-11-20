import openai
import glob
import backoff
import threading
import queue
import time

API_KEY = "sk-e8b81c39212e4a90a49ea7208db534a1"
API_ENDPOINT = "https://api.deepseek.com"

request_queue = queue.Queue()
response_queue = queue.Queue()


def request_worker():
    while True:
        prompt = request_queue.get()
        if prompt is None:  # 结束信号
            request_queue.task_done()  # 确保任务完成
            break
        try:
            response = send_and_request(prompt)
            response_queue.put(response)
        except Exception as e:
            print(f"请求处理出错: {e}")  # 处理异常
        finally:
            request_queue.task_done()  # 确保任务完成


@backoff.on_exception(wait_gen=backoff.expo, exception=openai.RateLimitError, max_tries=1)
def send_and_request(prompt):
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=API_ENDPOINT,
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system",
             "content": "你是AI播客助手，你会为用户在文章中提取安全，有帮助，准确的信息。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。你所提供的答案都应该由文章中的信息提供，不可以从外界获取信息。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content


def extract_blog_manuscript(article):
    prompt = f"""
    请将以下文章转换为播客文稿格式。提取主要观点,不允许引入外部知识和文章以外的信息，并且以适合演播的口吻和语气输出:

    "{article}"

    请控制输出内容按照博客的形式进行输出，按照一个播音员的口吻进行回复。
    """
    request_queue.put(prompt)  # 将请求放入队列
    response = response_queue.get()  # 从响应队列获取结果
    print(f"提取的文稿: {response}")  # 打印提取的文稿
    return response


def summary_blog(article):
    prompt = f"""
    请对以下文章进行简短的总结。提取主要观点，不允许引入外部知识和文章以外的信息。

    "{article}"

    你在回答中不需要对自己的角色进行描述，只需要进行文章内容总结。按照一整段进行输出，不要分段。
    """
    request_queue.put(prompt)  # 将请求放入队列
    response = response_queue.get()  # 从响应队列获取结果
    print(f"文章总结: {response}")  # 打印文章总结
    return response


def summary_to_manuscript(summaries):
    prompt = f"""
    请将以下文章总结提炼为播客文稿格式。确保提取主要观点，并以适合演播的口吻和语气输出:

    {''.join(summaries)}

    请控制输出内容按照博客的形式进行输出，按照一个播音员的口吻进行回复。

    """
    request_queue.put(prompt)  # 将请求放入队列
    response = response_queue.get()  # 从响应队列获取结果
    print(f"总结的文稿: {response}")  # 打印总结的文稿
    return response


def main():
    # 启动请求线程
    thread = threading.Thread(target=request_worker)
    thread.start()

    articles = []
    for filename in glob.glob('example*.txt'):
        with open(filename, 'r', encoding='utf-8') as file:
            articles.append(file.read())  # 读取整个文件内容
            print(f"读取文件: {filename}")

    if len(articles) == 1:
        manuscript = extract_blog_manuscript(articles[0].strip())
    else:
        summaries = []
        for article in articles:
            summary = summary_blog(article)
            summaries.append(summary)
        combined_article = summary_to_manuscript(summaries)
        manuscript = extract_blog_manuscript(combined_article.strip())

    request_queue.join()  # 确保所有请求都已处理
    request_queue.put(None)
    thread.join()
    print(f"最终文稿: {manuscript}")  # 输出API返回的内容
    return manuscript


if __name__ == "__main__":
    main()