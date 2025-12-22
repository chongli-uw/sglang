#!/usr/bin/bash


# format the script 

function send_one_request() {
  curl -X POST http://localhost:30000/generate \
       -H "Content-Type: application/json" \
       -d '{
            "text": "The capital city of France is",
            "stream": false,
            "return_logprob": true,
            "sampling_params": {
              "max_new_tokens": 10,
              "temperature": 0.0,
              "top_p": 1.0,
              "top_k": 1
            }
          }'

  echo ""
  sleep 1
}

send_one_request

curl http://localhost:30000/paras_configure_tp
sleep 1

send_one_request

