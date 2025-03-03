-- Enable the pgvector extension
create extension if not exists vector;

-- Create the memories table
create table if not exists hga_memories (
    id uuid default gen_random_uuid() primary key,
    key text not null,
    value jsonb not null,
    embedding vector(512),
    created_at timestamptz default now(),
    unique(key)
);

-- Create an index for vector similarity search
create index if not exists hga_memories_embedding_idx on hga_memories using ivfflat (embedding vector_cosine_ops);

-- Create a function for similarity search
create or replace function match_memories(
    query_embedding vector(512),
    match_threshold float,
    match_count int
)
returns table (
    key text,
    value jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        m.key,
        m.value,
        1 - (m.embedding <=> query_embedding) as similarity
    from hga_memories m
    where 1 - (m.embedding <=> query_embedding) > match_threshold
    order by m.embedding <=> query_embedding
    limit match_count;
end;
$$;
