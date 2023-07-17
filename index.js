import dotenv from 'dotenv';

import readline from 'readline';

import { ChatOpenAI } from 'langchain/chat_models/openai';
import { BufferMemory } from 'langchain/memory';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { HumanMessage, AIMessage } from 'langchain/schema';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { ChatMessageHistory } from 'langchain/memory';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const model = new ChatOpenAI({
  temperature: 0.9,
  openAIApiKey: OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
});

const CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. But the AI doesn't ask the human any question which are not related to the conversation.

Current conversation:
{chat_history}
Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
Human: {question}

AI:`;

export const makeChain = (vectorstore, model, chatHistory) => {
  const memory = new BufferMemory({
    memoryKey: 'chat_history',
    inputKey: 'question',
    chatHistory: chatHistory,
    returnMessages: true,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      memory,
      questionGeneratorChainOptions: {
        template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT,
        llm: model,
      },
    }
  );
  return chain;
};

const vectorStore = await MemoryVectorStore.fromExistingIndex(
  new OpenAIEmbeddings()
);

function jsonToConversation(json) {
  const conversation = [];
  for (let i = 0; i < json.length; i++) {
    if (json[i].role === 'user') {
      if (i === json.length - 1) {
        continue;
      }
      conversation.push(new HumanMessage(json[i].content));
    } else {
      conversation.push(new AIMessage(json[i].content));
    }
  }
  return conversation;
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const history = [];

export const query = async () => {
  rl.question('input:', async (question) => {
    question;

    history.push({ role: 'user', content: question });
    const conversation = jsonToConversation(history);
    const chatHistory = new ChatMessageHistory(conversation);

    const chain = makeChain(vectorStore, model, chatHistory);
    const sanitizedQuestion = question.trim().replaceAll('\n', ' ');
    const response = await chain.call({
      question: sanitizedQuestion,
    });
    console.log(response.text);
    history.push({ role: 'ai', content: response.text });
    query();
  });
};

query();
