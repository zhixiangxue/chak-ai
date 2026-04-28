"""
SQL: Built-in database query tool for chak

Provides four public methods exposed as LLM tools via NativeObjectTool:

    - query:   Execute a SELECT and return results as a formatted table
    - execute: Execute INSERT / UPDATE / DELETE / DDL
    - tables:  List all user tables in the database
    - schema:  Describe columns of a table

All methods take ``uri`` as their first argument so the LLM can work with
multiple databases in the same conversation.

Supported URI schemes:
    sqlite:///path/to/db.db           — SQLite (built-in, zero extra deps)
    postgresql://user:pass@host/db    — PostgreSQL (pip install psycopg2-binary)
    mysql://user:pass@host/db         — MySQL     (pip install pymysql)

Usage:
    from chak.tools.std import SQL
    sql = SQL()
    conv = Conversation(model, tools=[sql])
"""

from typing import Optional

_MAX_ROWS = 200        # default cap for SELECT results
_MAX_CELL_LEN = 300    # truncate very long cell values


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect(uri: str):
    """Return a DB-API 2.0 (conn, dialect) tuple for the given URI."""
    if uri.startswith("sqlite:///"):
        import sqlite3
        path = uri[len("sqlite:///"):]
        return sqlite3.connect(path), "sqlite"
    if uri in ("sqlite:///:memory:", ":memory:"):
        import sqlite3
        return sqlite3.connect(":memory:"), "sqlite"
    if uri.startswith("postgresql://") or uri.startswith("postgres://"):
        try:
            import psycopg2
        except ImportError:
            raise RuntimeError(
                "psycopg2 not installed — run `pip install psycopg2-binary`"
            )
        return psycopg2.connect(uri), "postgresql"
    if uri.startswith("mysql://"):
        try:
            import pymysql
        except ImportError:
            raise RuntimeError("pymysql not installed — run `pip install pymysql`")
        from urllib.parse import urlparse
        p = urlparse(uri)
        conn = pymysql.connect(
            host=p.hostname,
            port=p.port or 3306,
            user=p.username or "",
            password=p.password or "",
            database=(p.path or "").lstrip("/"),
            charset="utf8mb4",
        )
        return conn, "mysql"
    raise ValueError(
        f"Unsupported URI: {uri!r}. "
        "Use sqlite:///path.db, postgresql://user:pass@host/db, or mysql://user:pass@host/db"
    )


def _fmt_table(columns: list, rows: list, max_rows: int) -> str:
    """Format query results as a plain-text ASCII table."""
    if not columns:
        return "(no columns returned)"
    if not rows:
        return f"(0 rows)\nColumns: {', '.join(str(c) for c in columns)}"

    total = len(rows)
    display = rows[:max_rows]

    # Compute column widths
    widths = [min(len(str(c)), _MAX_CELL_LEN) for c in columns]
    for row in display:
        for i, val in enumerate(row):
            if i < len(widths):
                cell = "NULL" if val is None else str(val)[:_MAX_CELL_LEN]
                widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: list) -> str:
        parts = []
        for i, v in enumerate(vals):
            if i >= len(widths):
                break
            cell = "NULL" if v is None else str(v)[:_MAX_CELL_LEN]
            parts.append(cell.ljust(widths[i]))
        return " | ".join(parts)

    lines = [fmt_row(columns), "-+-".join("-" * w for w in widths)]
    lines += [fmt_row(r) for r in display]
    suffix = (
        f"\n\n(showing {max_rows} of {total}+ rows — use LIMIT/OFFSET to page)"
        if total > max_rows
        else f"\n\n({total} row{'s' if total != 1 else ''})"
    )
    return "\n".join(lines) + suffix


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

class SQL:
    """SQL database tool — query and modify SQLite, PostgreSQL, or MySQL databases.

    All methods accept ``uri`` as their first argument, allowing the LLM to
    target different databases within the same conversation.

    Supported URI formats::

        sqlite:///path/to/database.db
        postgresql://user:password@host:5432/dbname
        mysql://user:password@host:3306/dbname

    Example::

        sql = SQL()
        conv = Conversation(model, tools=[sql])
        await conv.asend("List the tables in sqlite:///shop.db and show me recent orders")
    """

    def __init__(self, max_rows: int = _MAX_ROWS):
        """
        Args:
            max_rows: Maximum rows returned by query() (default 200).
        """
        self._max_rows = max_rows

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def query(self, uri: str, sql: str, params: Optional[list] = None) -> str:
        """Execute a SQL SELECT statement and return results as a formatted table.

        Use for all read operations: SELECT, aggregations, joins, subqueries.
        Always use query() for SELECT — never execute().

        Args:
            uri:    Database URI.
                    SQLite:     ``sqlite:///path/to/db.db``
                    PostgreSQL: ``postgresql://user:pass@host/db``
                    MySQL:      ``mysql://user:pass@host/db``
            sql:    SELECT statement to execute.
            params: Optional list of positional parameters (``?`` for SQLite,
                    ``%s`` for PostgreSQL/MySQL) to avoid SQL injection.

        Returns:
            ASCII table of results with column headers, or an error string.
        """
        try:
            conn, _ = _connect(uri)
            try:
                cur = conn.cursor()
                cur.execute(sql, params or [])
                rows = cur.fetchmany(self._max_rows + 1)
                columns = [d[0] for d in (cur.description or [])]
            finally:
                conn.close()
            return _fmt_table(columns, rows, self._max_rows)
        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, uri: str, sql: str, params: Optional[list] = None) -> str:
        """Execute a write SQL statement: INSERT, UPDATE, DELETE, CREATE, DROP, etc.

        Do NOT use for SELECT — use query() instead.

        Args:
            uri:    Database URI (same format as query()).
            sql:    SQL statement to execute.
            params: Optional positional parameters.

        Returns:
            Summary of rows affected, or an error string.
        """
        try:
            conn, _ = _connect(uri)
            try:
                cur = conn.cursor()
                cur.execute(sql, params or [])
                conn.commit()
                affected = cur.rowcount
            finally:
                conn.close()
            return f"OK — {affected} row{'s' if affected != 1 else ''} affected"
        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # tables
    # ------------------------------------------------------------------

    def tables(self, uri: str) -> str:
        """List all user tables in the database.

        Always call this first when exploring an unfamiliar database.

        Args:
            uri: Database URI.

        Returns:
            Newline-separated list of table names, or an error string.
        """
        try:
            conn, dialect = _connect(uri)
            try:
                cur = conn.cursor()
                if dialect == "sqlite":
                    cur.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                elif dialect == "postgresql":
                    cur.execute(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = 'public' ORDER BY tablename"
                    )
                else:  # mysql
                    cur.execute("SHOW TABLES")
                rows = cur.fetchall()
            finally:
                conn.close()
            if not rows:
                return "(no tables found)"
            return "\n".join(r[0] for r in rows)
        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    def schema(self, uri: str, table: str) -> str:
        """Describe the columns of a table.

        Use after tables() to understand the data model before writing queries.

        Args:
            uri:   Database URI.
            table: Table name.

        Returns:
            Column definitions (name, type, nullable, default), or an error string.
        """
        try:
            conn, dialect = _connect(uri)
            try:
                cur = conn.cursor()
                if dialect == "sqlite":
                    cur.execute(f"PRAGMA table_info({table})")
                    rows = cur.fetchall()
                    if not rows:
                        return f"Error: table '{table}' not found"
                    return _fmt_table(
                        ["cid", "name", "type", "notnull", "dflt_value", "pk"],
                        rows,
                        200,
                    )
                elif dialect == "postgresql":
                    cur.execute(
                        """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = %s ORDER BY ordinal_position
                        """,
                        (table,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return f"Error: table '{table}' not found"
                    return _fmt_table(
                        ["column_name", "data_type", "is_nullable", "column_default"],
                        rows, 200,
                    )
                else:  # mysql
                    cur.execute(f"DESCRIBE `{table}`")
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return _fmt_table(cols, rows, 200)
            finally:
                conn.close()
        except Exception as e:
            return f"Error: {e}"

    def __repr__(self) -> str:
        return f"<SQL max_rows={self._max_rows}>"
