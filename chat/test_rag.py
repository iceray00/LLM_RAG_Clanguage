from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import Pipeline

# no parameter init, we don't use any runtime template variables
prompt_builder = ChatPromptBuilder()
generator = OllamaChatGenerator(model="zephyr",
                            url = "http://localhost:11434",
                            generation_kwargs={
                              "temperature": 0.9,
                              })

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", generator)
pipe.connect("prompt_builder.prompt", "llm.messages")
location = "Berlin"
messages = [ChatMessage.from_system("Always respond in Spanish even if some input data is in other languages."),
            ChatMessage.from_user("Tell me about {{location}}")]
print(pipe.run(data={"prompt_builder": {"template_variables":{"location": location}, "template": messages}}))


print("\n")
print(messages)


# root@autodl-container-a63e40be4b-9396484e:~/iceray/chat# python3 test_rag.py
# {'llm': {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='柏林，德国的首都和最大城市，位于德国东部，拥有丰富的历史、文化和艺术景观。这座城市因其独特的魅力而著名，并且在第二次世界大战后重建的过程中发展成为一个充满活力和现代感的城市。\n\n柏林的历史非常丰富，它见证了普鲁士帝国、德意志帝国、魏玛共和国、纳粹统治以及冷战时期的东西方分裂。1989年墙的倒塌标志着东西德的统一和全球冷战格局的重要转变。在城市中可以看到许多反映这一历史变迁的标志物，比如勃兰登堡门（Brandenburg Gate）、查理检查站（Checkpoint Charlie）等。\n\n柏林不仅是政治中心，也是文化和艺术的汇聚地。它拥有世界一流的博物馆、画廊、音乐厅和剧院，如柏林国家博物馆、波茨坦广场的艺术论坛和德意志歌剧院。城市还以其夜生活著称，包括多种类型的俱乐部和酒吧。\n\n此外，柏林也因其绿色空间而著名，拥有广阔的公园和林地，比如夏洛特湖（Lake Charlottenburg）和普伦茨劳贝格公园（Prenzlauer Berg Park），为市民和游客提供了一个逃离城市喧嚣的宁静之地。\n\n柏林也是欧洲的文化、媒体和技术中心之一。许多国际机构和跨国公司都在此设有总部或重要分支，吸引了大量来自世界各地的人才和创意产业。在现代柏林，您可以看到古老的建筑与最新的科技和设计并存，展现了这个城市独特的融合性和创新精神。\n\n总体而言，柏林是一个多面体的城市，结合了历史的深度、文化的活力、艺术的先锋性以及现代化的生活方式，使其成为探索德国乃至欧洲的理想目的地。')], _name=None, _meta={'model': 'qwen2', 'created_at': '2025-02-07T05:12:21.264912953Z', 'done': True, 'done_reason': 'stop', 'total_duration': 3889761318, 'load_duration': 35539279, 'prompt_eval_count': 31, 'prompt_eval_duration': 22000000, 'eval_count': 345, 'eval_duration': 3815000000})]}}
#
#
# [ChatMessage(_role=<ChatRole.SYSTEM: 'system'>, _content=[TextContent(text='Always respond in Chinese even if some input data is in other languages.')], _name=None, _meta={}), ChatMessage(_role=<ChatRole.USER: 'user'>, _content=[TextContent(text='Tell me about {{location}}')], _name=None, _meta={})]
