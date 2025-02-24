def generate_project_guidance(project_description):
    # 示例代码片段
    sample_code_snippets = """
    ### 学生成绩管理系统 - 文件读取部分
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #define LEN sizeof(struct Student)
    struct Student 
    {
            int clas;
            int num;
            char name[20];
            char cou[20];
            int score;
            struct Student *next;
    };

    struct Student *creat(void)
    {
            struct Student *head;
            struct Student *p1,*p2;
            int n=0;
            printf("班级 学号 姓名 课程 得分(输入0 0 0 0 0结束)：\n");
            p1=p2=(struct Student*)malloc(LEN);
            scanf("%d%d%s%s%d",&p1->clas,&p1->num,p1->name,p1->cou,&p1->score);
            head=NULL;
            while(p1->num!=0)
            {
                    n=n+1;
                    if(n==1)head=p1;
                    else p2->next=p1;
                    p2=p1;
                    p1=(struct Student*)malloc(LEN);
                    scanf("%d%d%s%s%d",&p1->clas,&p1->num,p1->name,p1->cou,&p1->score);
            }
            p2->next=NULL;
            return(head);
    }    
    """

    prompt = f"""
    根据以下提供的C语言课程大作业描述，请结合下面给出的几个示例代码片段，提供一份详细的指导方案。
    如果描述中存在模糊性或缺少细节，请基于示例代码和常见需求进行合理推测，并选择最接近的示例作为参考。

    输出应遵循以下格式：

    # 项目概述
    - 目标：简述项目的主要目标。

    # 任务分解
    - 步骤1：描述第一步及其目的。针对步骤1提出实现建议，并可引用或者修改完善示例代码中的相关部分。

    # 挑战与解决方案
    - 挑战1：列出可能遇到的问题及解决方案；提出代码中某些部分可以改进或者使用的替代实现方式

    # 参考资料
    - 资源1：提供有助于完成项目的资源链接或书籍章节。

    下面是常见的学生信息管理系统示例代码片段供参考：
    {sample_code_snippets}

    大作业问题描述：
    {project_description}

    注意：请依据具体的作业要求调整上述模板，必要时可以对示例代码进行适当修改以更好地匹配当前项目的需求。
    """
    # response = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=prompt,
    #     max_tokens=800
    # )
    # return response.choices[0].text.strip()


# Others

def generate_project_guidance(project_description):
    # 几个不同类型的示例代码片段
    sample_code_snippets = """
    ### 学生成绩管理系统 - 文件读取部分
    ```c
    FILE *file;
    struct Student { char name[50]; int id; float gpa; };
    file = fopen("students.txt", "r");
    if (file == NULL) { printf("无法打开文件\n"); return 1; }
    ```

    ### 图书管理系统 - 添加书籍功能
    ```c
    void addBook() {
        Book newBook;
        printf("请输入书名: ");
        scanf("%s", newBook.title);
        // 省略其他输入及保存逻辑...
    }
    ```

    ### 简单文本编辑器 - 打开文件功能
    ```c
    void openFile(char fileName[]) {
        FILE *file = fopen(fileName, "r");
        if (file == NULL) {
            printf("文件未找到。\n");
            return;
        }
        // 处理文件内容...
    }
    ```
    """

    prompt = f"""
    根据以下提供的C语言课程大作业描述，请结合下面给出的几个示例代码片段，提供一份详细的指导方案。
    如果描述中存在模糊性或缺少细节，请基于示例代码和常见需求进行合理推测，并选择最接近的示例作为参考。

    输出应遵循以下格式：

    # 项目概述
    - 目标：简述项目的主要目标。

    # 任务分解
    - 步骤1：描述第一步及其目的。针对步骤1提出实现建议，并可引用或者修改完善示例代码中的相关部分。

    # 挑战与解决方案
    - 挑战1：列出可能遇到的问题及解决方案；提出代码中某些部分可以改进或者使用的替代实现方式

    # 参考资料
    - 资源1：提供有助于完成项目的资源链接或书籍章节。

    下面是一些常见的C语言大作业示例代码片段供参考：
    {sample_code_snippets}

    大作业描述：
    {project_description}

    注意：请依据具体的作业要求调整上述模板，必要时可以对示例代码进行适当修改以更好地匹配当前项目的需求。
    """
    # response = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=prompt,
    #     max_tokens=800
    # )
    # return response.choices[0].text.strip()



