struct Variable;

extern void jitc_cuda_render_queue_send(Variable *v,
                                        Variable *queue_buf,
                                        Variable *msg_type);

extern void jitc_cuda_render_queue_recv(Variable *v,
                                        Variable *queue_ticket);
