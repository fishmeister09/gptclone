import dotenv from 'dotenv';

import readline from 'readline';

import { ChatOpenAI } from 'langchain/chat_models/openai';
import { BufferMemory } from 'langchain/memory';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { HumanMessage, AIMessage } from 'langchain/schema';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { ChatMessageHistory } from 'langchain/memory';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import prompts from './configs.json' assert { type: 'json' };

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const model = new ChatOpenAI({
  temperature: 0.8,
  openAIApiKey: OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
});
let QUESTION_GENERATOR_PROMPT;
let QA_PROMPT;
function getPrompt() {
  QUESTION_GENERATOR_PROMPT = prompts.QUESTION_GENERATOR_PROMPT;
  QA_PROMPT = prompts.QA_PROMPT;

  // get bot_metadata
  const bot_metadata = prompts.bot_metaData;
  QA_PROMPT = QA_PROMPT.replace('{bot_style}', bot_metadata.bot_style);
  QA_PROMPT = QA_PROMPT.replace('{bot_role}', bot_metadata.bot_role);
  QA_PROMPT = QA_PROMPT.replace('{bot_name}', bot_metadata.bot_name);
  QA_PROMPT = QA_PROMPT.replace('{bot_tone}', bot_metadata.bot_tone);
}

function convertChatHistoryToLanchainMemory(chatHistory) {
  let pastMessages = [];

  for (let i = 0; i < chatHistory.length; i++) {
    if (chatHistory[i].role === 'user') {
      if (i === chatHistory.length - 1) {
        continue;
      }
      pastMessages.push(new HumanMessage(chatHistory[i].content));
    } else {
      pastMessages.push(new AIMessage(chatHistory[i].content));
    }
  }
  const memory = new BufferMemory({
    chatHistory: new ChatMessageHistory(pastMessages),
    memoryKey: 'chat_history',
    inputKey: 'question',
    outputKey: 'text',
    returnMessages: true,
  });

  return memory;
}

export const makeChain = (vectorstore, model, chatHistory) => {
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      memory: convertChatHistoryToLanchainMemory(chatHistory),
      // qaChainOptions: {
      //   type: 'stuff', // Chain Type
      //   prompt: PromptTemplate.fromTemplate(QA_PROMPT),
      // },
      // questionGeneratorChainOptions: {
      //   template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT,
      //   llm: nonStreamingModel,
      // },
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: QUESTION_GENERATOR_PROMPT,
    }
  );
  return chain;
};

const vectorStore = await MemoryVectorStore.fromExistingIndex(
  new OpenAIEmbeddings()
);

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const history = [];

export const query = async () => {
  getPrompt();
  rl.question('input:', async (question) => {
    question;

    history.push({ role: 'user', content: question });

    const chain = makeChain(vectorStore, model, history);
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
